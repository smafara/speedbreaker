// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// Dear ImGui's draw data on the RHI (gpu/rhi): the UI on a backend that has
// no Dear ImGui backend of its own here (Metal). Vertices are read by the
// vertex shader from a storage buffer (the RHI has no vertex input), the
// font atlas and any other texture ImGui asks for are RHI images, and the
// recording is the RHI's, so it is the same code on every backend.
//
// A frame is two calls on one command list: Prepare, outside a pass (texture
// uploads, the frame's vertices and indices), then Draw, inside the pass that
// writes the target. Buffers are kept for three frames (the presenter has two
// in flight), so a slot is rewritten only after its last use finished.
#pragma once
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include <gpu/rhi/command_list.h>
#include <gpu/rhi/device.h>

struct ImDrawData;

namespace ui
{
    class RhiRenderer
    {
    public:
        explicit RhiRenderer(gpu::rhi::Device& device);
        ~RhiRenderer();

        // False: the shaders or the pipeline couldn't be made (no UI).
        bool Ready() const { return m_ready; }

        // Outside a pass. `target`: the format of the image Draw will write.
        void Prepare(gpu::rhi::CommandList& list, ImDrawData* data, gpu::rhi::Format target);
        // Inside the pass on that image.
        void Draw(gpu::rhi::CommandList& list, ImDrawData* data);

    private:
        struct Texture
        {
            std::unique_ptr<gpu::rhi::Image> image;
            uint32_t width = 0, height = 0;
        };
        struct Slot
        {
            std::unique_ptr<gpu::rhi::Buffer> vertices, indices;
            uint64_t vertexBytes = 0, indexBytes = 0;
            std::vector<std::unique_ptr<gpu::rhi::Buffer>> staging;  // texture uploads of this frame
        };
        static constexpr uint32_t kSlots = 3;

        bool CreatePipeline(gpu::rhi::Format target);
        void UpdateTextures(gpu::rhi::CommandList& list, ImDrawData* data, Slot& slot);

        gpu::rhi::Device& m_device;
        bool m_ready = false;
        std::unique_ptr<gpu::rhi::Sampler> m_sampler;
        std::unique_ptr<gpu::rhi::BindingLayout> m_layout;
        std::unique_ptr<gpu::rhi::Pipeline> m_pipeline;
        gpu::rhi::Format m_format = gpu::rhi::Format::Undefined;
        std::vector<uint32_t> m_vertexSpirv, m_fragmentSpirv;
        Slot m_slots[kSlots];
        uint32_t m_frame = 0;
        // Textures ImGui let go of, kept until the GPU can't be using them.
        struct Dying { std::unique_ptr<Texture> texture; uint32_t frame; };
        std::deque<Dying> m_dying;
        std::unordered_map<Texture*, std::unique_ptr<Texture>> m_textures;  // the live ones (ImTextureData::TexID points at one)
    };
}
