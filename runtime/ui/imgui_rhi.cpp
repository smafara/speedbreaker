// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING). See imgui_rhi.h.
#include <stdafx.h>
#include "imgui_rhi.h"

#include <gpu/renderer.h>

#include <imgui.h>

#include <algorithm>
#include <cstring>

namespace ui
{
    using namespace gpu::rhi;

    namespace
    {
        // One vertex is an ImDrawVert: pos (2 floats), uv (2 floats), colour (packed ABGR): 5 words.
        const char* kVertexGlsl = R"(#version 460
layout(std430, set = 0, binding = 0) readonly buffer Vertices { uint v[]; };
layout(push_constant) uniform P { vec2 scale; vec2 translate; uint base; } p;
layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 color;
void main()
{
    uint i = (p.base + uint(gl_VertexIndex)) * 5u;
    vec2 pos = vec2(uintBitsToFloat(v[i]), uintBitsToFloat(v[i + 1u]));
    uv = vec2(uintBitsToFloat(v[i + 2u]), uintBitsToFloat(v[i + 3u]));
    color = unpackUnorm4x8(v[i + 4u]);
    gl_Position = vec4(pos * p.scale + p.translate, 0.0, 1.0);
}
)";
        const char* kFragmentGlsl = R"(#version 460
layout(set = 0, binding = 1) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = color * texture(tex, uv);
}
)";
        struct Constants
        {
            float scale[2], translate[2];
            uint32_t base;
        };
    }

    RhiRenderer::RhiRenderer(Device& device) : m_device(device)
    {
        std::string log;
        m_vertexSpirv = gpu::renderer::CompileShader(kVertexGlsl, 0, log);
        if (m_vertexSpirv.empty())
        {
            fprintf(stderr, "[ui] vertex shader failed:\n%s\n", log.c_str());
            return;
        }
        m_fragmentSpirv = gpu::renderer::CompileShader(kFragmentGlsl, 1, log);
        if (m_fragmentSpirv.empty())
        {
            fprintf(stderr, "[ui] fragment shader failed:\n%s\n", log.c_str());
            return;
        }
        SamplerDesc sd;
        sd.u = sd.v = sd.w = AddressMode::ClampToEdge;
        sd.maxLod = 0.0f;
        m_sampler = device.CreateSampler(sd);
        BindingLayoutDesc ld;
        ld.tables.push_back({ { { 0, BindingKind::StorageBuffer, uint8_t(ShaderStage::Vertex) },
                                { 1, BindingKind::CombinedImageSampler, uint8_t(ShaderStage::Fragment), m_sampler.get() } },
            true });
        ld.pushConstantBytes = sizeof(Constants);
        m_layout = device.CreateBindingLayout(ld);
        m_ready = true;
    }

    RhiRenderer::~RhiRenderer() = default;

    bool RhiRenderer::CreatePipeline(Format target)
    {
        GraphicsPipelineDesc d;
        d.vertex = { m_vertexSpirv.data(), m_vertexSpirv.size() };
        d.fragment = { m_fragmentSpirv.data(), m_fragmentSpirv.size() };
        d.layout = m_layout.get();
        d.colorFormats[0] = target;
        d.blend[0] = { true, BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendOp::Add, BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha, BlendOp::Add, 0xF };
        m_pipeline = m_device.CreateGraphicsPipeline(d);
        m_format = m_pipeline ? target : Format::Undefined;
        if (!m_pipeline)
            fprintf(stderr, "[ui] the pipeline couldn't be made\n");
        return m_pipeline != nullptr;
    }

    // Creates, updates and retires the textures ImGui asks for (its font atlas, mainly).
    void RhiRenderer::UpdateTextures(CommandList& list, ImDrawData* data, Slot& slot)
    {
        if (!data->Textures)
            return;
        for (ImTextureData* tex : *data->Textures)
        {
            if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames >= int(kSlots))
            {
                if (auto* texture = reinterpret_cast<Texture*>(tex->TexID))
                    if (auto it = m_textures.find(texture); it != m_textures.end())
                    {
                        m_dying.push_back({ std::move(it->second), m_frame });
                        m_textures.erase(it);
                    }
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_Destroyed);
                continue;
            }
            if (tex->Status != ImTextureStatus_WantCreate && tex->Status != ImTextureStatus_WantUpdates)
                continue;
            const bool create = tex->Status == ImTextureStatus_WantCreate;
            Texture* texture = nullptr;
            if (create)
            {
                auto fresh = std::make_unique<Texture>();
                ImageDesc id;
                id.format = Format::RGBA8;
                id.extent = { uint32_t(tex->Width), uint32_t(tex->Height), 1 };
                id.usage = kImageSampled | kImageCopyDst;
                id.name = "ui texture";
                fresh->image = m_device.CreateImage(id);
                if (!fresh->image)
                {
                    fprintf(stderr, "[ui] a %dx%d texture couldn't be made\n", tex->Width, tex->Height);
                    continue;
                }
                fresh->width = uint32_t(tex->Width);
                fresh->height = uint32_t(tex->Height);
                texture = fresh.get();
                m_textures[texture] = std::move(fresh);
                tex->SetTexID(ImTextureID(reinterpret_cast<uintptr_t>(texture)));
            }
            else
                texture = reinterpret_cast<Texture*>(tex->TexID);
            if (!texture)
                continue;
            // The rectangle to send: all of a new texture, else what changed.
            ImTextureRect r = create ? ImTextureRect{ 0, 0, uint16_t(tex->Width), uint16_t(tex->Height) } : tex->UpdateRect;
            BufferDesc bd;
            bd.size = uint64_t(r.w) * r.h * 4;
            bd.usage = kBufferCopySrc;
            bd.memory = MemoryKind::Upload;
            bd.name = "ui texture upload";
            std::unique_ptr<Buffer> staging = m_device.CreateBuffer(bd);
            if (!staging)
                continue;
            auto* out = static_cast<uint8_t*>(staging->Map());
            for (int y = 0; y < r.h; y++)
            {
                const uint8_t* row = static_cast<const uint8_t*>(tex->GetPixelsAt(r.x, r.y + y));
                uint8_t* dst = out + size_t(y) * r.w * 4;
                if (tex->Format == ImTextureFormat_RGBA32)
                    memcpy(dst, row, size_t(r.w) * 4);
                else  // Alpha8: white with that alpha
                    for (int x = 0; x < r.w; x++)
                    {
                        dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = 255;
                        dst[x * 4 + 3] = row[x];
                    }
            }
            ImageTransition before{ texture->image.get(), create ? State::Undefined : State::ShaderRead, State::CopyDst };
            list.Transition(&before, 1);
            BufferImageCopy copy;
            copy.region = { r.x, r.y, r.w, r.h };
            list.CopyBufferToImage(*staging, *texture->image, &copy, 1);
            ImageTransition after{ texture->image.get(), State::CopyDst, State::ShaderRead };
            list.Transition(&after, 1);
            slot.staging.push_back(std::move(staging));
            tex->SetStatus(ImTextureStatus_OK);
        }
    }

    void RhiRenderer::Prepare(CommandList& list, ImDrawData* data, Format target)
    {
        if (!m_ready || !data)
            return;
        m_frame++;
        Slot& slot = m_slots[m_frame % kSlots];
        slot.staging.clear();  // (this slot's last frame is done)
        while (!m_dying.empty() && m_frame - m_dying.front().frame >= kSlots)
            m_dying.pop_front();
        if (target != m_format && !CreatePipeline(target))
            return;
        UpdateTextures(list, data, slot);

        uint64_t vertexBytes = uint64_t(data->TotalVtxCount) * sizeof(ImDrawVert);
        uint64_t indexBytes = uint64_t(data->TotalIdxCount) * sizeof(ImDrawIdx);
        if (!vertexBytes || !indexBytes)
            return;
        auto grow = [&](std::unique_ptr<Buffer>& buffer, uint64_t& have, uint64_t need, uint32_t usage, const char* name) {
            if (buffer && have >= need)
                return;
            BufferDesc bd;
            bd.size = std::max<uint64_t>(need + need / 2, 64 * 1024);
            bd.usage = usage;
            bd.memory = MemoryKind::Upload;
            bd.name = name;
            buffer = m_device.CreateBuffer(bd);
            have = buffer ? bd.size : 0;
        };
        grow(slot.vertices, slot.vertexBytes, vertexBytes, kBufferStorage, "ui vertices");
        grow(slot.indices, slot.indexBytes, indexBytes, kBufferIndex, "ui indices");
        if (!slot.vertices || !slot.indices)
            return;
        auto* vtx = static_cast<uint8_t*>(slot.vertices->Map());
        auto* idx = static_cast<uint8_t*>(slot.indices->Map());
        for (const ImDrawList* cl : data->CmdLists)
        {
            memcpy(vtx, cl->VtxBuffer.Data, size_t(cl->VtxBuffer.Size) * sizeof(ImDrawVert));
            memcpy(idx, cl->IdxBuffer.Data, size_t(cl->IdxBuffer.Size) * sizeof(ImDrawIdx));
            vtx += size_t(cl->VtxBuffer.Size) * sizeof(ImDrawVert);
            idx += size_t(cl->IdxBuffer.Size) * sizeof(ImDrawIdx);
        }
    }

    void RhiRenderer::Draw(CommandList& list, ImDrawData* data)
    {
        Slot& slot = m_slots[m_frame % kSlots];
        if (!m_ready || !m_pipeline || !data || !slot.vertices || !slot.indices || !data->TotalVtxCount)
            return;
        const float fbW = data->DisplaySize.x * data->FramebufferScale.x, fbH = data->DisplaySize.y * data->FramebufferScale.y;
        if (fbW <= 0 || fbH <= 0)
            return;
        list.SetPipeline(*m_pipeline);
        list.SetViewport(0, 0, fbW, fbH, 0.0f, 1.0f);
        list.BindIndexBuffer(*slot.indices, 0, sizeof(ImDrawIdx) == 4);
        Constants c;
        c.scale[0] = 2.0f / data->DisplaySize.x;
        c.scale[1] = 2.0f / data->DisplaySize.y;
        c.translate[0] = -1.0f - data->DisplayPos.x * c.scale[0];
        c.translate[1] = -1.0f - data->DisplayPos.y * c.scale[1];
        const ImVec2 off = data->DisplayPos, scale = data->FramebufferScale;
        uint32_t vertexBase = 0, indexBase = 0;
        Texture* bound = nullptr;
        for (const ImDrawList* cl : data->CmdLists)
        {
            for (const ImDrawCmd& cmd : cl->CmdBuffer)
            {
                if (cmd.UserCallback)
                    continue;  // (none are used here)
                ImVec2 min((cmd.ClipRect.x - off.x) * scale.x, (cmd.ClipRect.y - off.y) * scale.y);
                ImVec2 max((cmd.ClipRect.z - off.x) * scale.x, (cmd.ClipRect.w - off.y) * scale.y);
                min.x = std::clamp(min.x, 0.0f, fbW);
                min.y = std::clamp(min.y, 0.0f, fbH);
                max.x = std::clamp(max.x, 0.0f, fbW);
                max.y = std::clamp(max.y, 0.0f, fbH);
                if (max.x <= min.x || max.y <= min.y)
                    continue;
                list.SetScissor({ int32_t(min.x), int32_t(min.y), uint32_t(max.x - min.x), uint32_t(max.y - min.y) });
                auto* texture = reinterpret_cast<Texture*>(cmd.GetTexID());
                if (!texture || !m_textures.contains(texture))
                    continue;
                if (texture != bound)
                {
                    BindingEntry entries[2] = {};
                    entries[0].slot = 0;
                    entries[0].buffer = { slot.vertices.get(), 0, ~0ull };
                    entries[1].slot = 1;
                    entries[1].image = { texture->image.get(), nullptr, nullptr };
                    list.PushBindings(*m_layout, 0, entries, 2);
                    bound = texture;
                }
                c.base = vertexBase + cmd.VtxOffset;
                list.PushConstants(*m_layout, &c, sizeof(c));
                list.DrawIndexed(cmd.ElemCount, 1, indexBase + cmd.IdxOffset, 0, 0);
            }
            vertexBase += uint32_t(cl->VtxBuffer.Size);
            indexBase += uint32_t(cl->IdxBuffer.Size);
        }
    }
}
