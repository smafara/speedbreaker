// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
// SPIR-V to MSL (SPIRV-Cross) and MSL to a Metal function; see metal_device.cpp.
#include "metal_internal.h"

#include <spirv_cross/spirv_msl.hpp>

#include <cstring>

namespace gpu::rhi::metal
{
    namespace impl
    {
        namespace
        {
            // MSL for `code` as stage `model` against `layout`: SPIRV-Cross's.
            std::string Translate(const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, std::string& entry,
                MTL::Size* size, bool pointSize)
            {
                spirv_cross::CompilerMSL compiler(code.spirv, code.words);
                spirv_cross::CompilerMSL::Options options;
                options.platform = spirv_cross::CompilerMSL::Options::macOS;
                options.set_msl_version(3, 0);
                options.argument_buffers = false;
                options.enable_point_size_builtin = pointSize;  // (Metal rejects a point size unless it draws points)
                options.enable_decoration_binding = false;
                // Vulkan's clip space has Y down and Metal's up: the vertex shaders'
                // gl_Position.y is flipped, so a picture comes out the same way up.
                options.pad_fragment_output_components = true;
                compiler.set_msl_options(options);
                spirv_cross::CompilerGLSL::Options common = compiler.get_common_options();
                common.vertex.flip_vert_y = model == spv::ExecutionModelVertex;
                compiler.set_common_options(common);
                for (const SlotMap& m : layout.Slots())
                {
                    spirv_cross::MSLResourceBinding b;
                    b.stage = model;
                    b.desc_set = m.table;
                    b.binding = m.slot.index;
                    b.msl_buffer = m.buffer != ~0u ? m.buffer : 0;
                    b.msl_texture = m.texture != ~0u ? m.texture : 0;
                    b.msl_sampler = m.sampler != ~0u ? m.sampler : 0;
                    compiler.add_msl_resource_binding(b);
                }
                if (layout.Desc().pushConstantBytes)
                {
                    spirv_cross::MSLResourceBinding b;
                    b.stage = model;
                    b.desc_set = spirv_cross::kPushConstDescSet;
                    b.binding = spirv_cross::kPushConstBinding;
                    b.msl_buffer = layout.PushConstantBuffer();
                    compiler.add_msl_resource_binding(b);
                }
                std::string msl = compiler.compile();
                entry = compiler.get_cleansed_entry_point_name("main", model);
                if (size)
                {
                    const spirv_cross::SPIREntryPoint& ep = compiler.get_entry_point("main", model);
                    *size = MTL::Size::Make(ep.workgroup_size.x, ep.workgroup_size.y, ep.workgroup_size.z);
                }
                return msl;
            }

            // Two 64-bit FNV-1a runs (different primes and starts): 128 bits, so two shaders sharing
            // a key is not a thing to plan for.
            class Hasher
            {
            public:
                void Add(const void* data, size_t size)
                {
                    const auto* bytes = static_cast<const uint8_t*>(data);
                    for (size_t i = 0; i < size; i++)
                    {
                        m_a = (m_a ^ bytes[i]) * 0x100000001b3ull;
                        m_b = (m_b ^ bytes[i]) * 0x9e3779b97f4a7c15ull;
                    }
                }
                template <class T> void Add(const T& value) { Add(&value, sizeof(value)); }
                MslCache::Key Key() const { return { m_a, m_b }; }

            private:
                uint64_t m_a = 0xcbf29ce484222325ull, m_b = 0x6c62272e07bb0142ull;
            };

#ifndef SB_SPIRV_CROSS_ID
#define SB_SPIRV_CROSS_ID ""
#endif
            // Bump when Translate's options change (they are not in the SPIR-V, the layout or the library).
            constexpr char kTranslateVersion[] = "msl-1";

            MslCache::Key KeyOf(const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, bool pointSize)
            {
                Hasher h;
                h.Add(kTranslateVersion, sizeof(kTranslateVersion));
                h.Add(SB_SPIRV_CROSS_ID, sizeof(SB_SPIRV_CROSS_ID));
                h.Add(uint32_t(model));
                h.Add(uint8_t(pointSize));
                h.Add(layout.Desc().pushConstantBytes);
                h.Add(layout.PushConstantBuffer());
                for (const SlotMap& m : layout.Slots())
                {
                    h.Add(m.table);
                    h.Add(m.slot.index);
                    h.Add(m.buffer);
                    h.Add(m.texture);
                    h.Add(m.sampler);
                }
                h.Add(uint64_t(code.words));
                h.Add(code.spirv, code.words * sizeof(uint32_t));
                return h.Key();
            }

            constexpr char kFileMagic[8] = { 'S', 'B', 'M', 'S', 'L', 'C', '0', '1' };
        }

        std::string ToMsl(MslCache& cache, const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, std::string& entry,
            MTL::Size* size, bool pointSize)
        {
            const MslCache::Key key = KeyOf(code, model, layout, pointSize);
            MslCache::Code found;
            if (cache.Find(key, found))
            {
                entry = std::move(found.entry);
                if (size)
                    *size = MTL::Size::Make(found.group[0], found.group[1], found.group[2]);
                return std::move(found.msl);
            }
            MTL::Size group{};
            std::string msl = Translate(code, model, layout, entry, &group, pointSize);
            if (size)
                *size = group;
            cache.Add(key, { msl, entry, { uint32_t(group.width), uint32_t(group.height), uint32_t(group.depth) } });
            return msl;
        }

        bool MslCache::Find(const Key& key, Code& code) const
        {
            std::lock_guard lock(m_mutex);
            const auto it = m_entries.find(key);
            if (it == m_entries.end())
                return false;
            code = it->second;
            return true;
        }

        void MslCache::Add(const Key& key, Code code)
        {
            std::lock_guard lock(m_mutex);
            m_entries.emplace(key, std::move(code));
        }

        size_t MslCache::Size() const
        {
            std::lock_guard lock(m_mutex);
            return m_entries.size();
        }

        // The file: the magic, an entry count (u32), then per entry the key (2 x u64), the entry point's
        // and the MSL's lengths (u32 each), the workgroup size (3 x u32), the two strings. Host byte order.
        std::vector<uint8_t> MslCache::Save() const
        {
            std::lock_guard lock(m_mutex);
            if (m_entries.empty())
                return {};
            std::vector<uint8_t> out(kFileMagic, kFileMagic + sizeof(kFileMagic));
            auto put = [&](const void* data, size_t size) {
                const auto* bytes = static_cast<const uint8_t*>(data);
                out.insert(out.end(), bytes, bytes + size);
            };
            const uint32_t count = uint32_t(m_entries.size());
            put(&count, sizeof(count));
            for (const auto& [key, code] : m_entries)
            {
                const uint32_t lengths[2] = { uint32_t(code.entry.size()), uint32_t(code.msl.size()) };
                put(&key.a, sizeof(key.a));
                put(&key.b, sizeof(key.b));
                put(lengths, sizeof(lengths));
                put(code.group, sizeof(code.group));
                put(code.entry.data(), code.entry.size());
                put(code.msl.data(), code.msl.size());
            }
            return out;
        }

        size_t MslCache::Load(const void* data, size_t size)
        {
            const auto* bytes = static_cast<const uint8_t*>(data);
            size_t at = 0;
            auto take = [&](void* to, size_t n) {
                if (n > size - at)
                    return false;
                std::memcpy(to, bytes + at, n);
                at += n;
                return true;
            };
            char magic[sizeof(kFileMagic)];
            uint32_t count = 0;
            if (!data || !take(magic, sizeof(magic)) || std::memcmp(magic, kFileMagic, sizeof(magic)) != 0 || !take(&count, sizeof(count)))
                return 0;
            std::map<Key, Code> loaded;
            for (uint32_t i = 0; i < count; i++)
            {
                Key key;
                uint32_t lengths[2];
                Code code;
                if (!take(&key.a, sizeof(key.a)) || !take(&key.b, sizeof(key.b)) || !take(lengths, sizeof(lengths)) ||
                    !take(code.group, sizeof(code.group)) || lengths[0] > size - at || lengths[1] > size - at - lengths[0])
                    return 0;  // (cut short or not ours: all of it or none)
                code.entry.assign(reinterpret_cast<const char*>(bytes + at), lengths[0]);
                at += lengths[0];
                code.msl.assign(reinterpret_cast<const char*>(bytes + at), lengths[1]);
                at += lengths[1];
                loaded.emplace(key, std::move(code));
            }
            std::lock_guard lock(m_mutex);
            for (auto& [key, code] : loaded)
                m_entries.emplace(key, std::move(code));
            return loaded.size();
        }

        // A function of `msl` named `entry`, or an empty pointer (logged).
        Ptr<MTL::Function> MakeFunction(MTL::Device& device, const std::string& msl, const std::string& entry)
        {
            Pool pool;
            auto options = Own(MTL::CompileOptions::alloc()->init());
            options->setLanguageVersion(MTL::LanguageVersion3_0);
            // Fast math makes world frames differ from MoltenVK's by up to 37 levels
            // (mean 0.016); safe math, by up to 11 (mean 0.0003).
            options->setMathMode(MTL::MathModeSafe);
            NS::Error* error = nullptr;
            NS::String* source = NS::String::string(msl.c_str(), NS::UTF8StringEncoding);
            Ptr<MTL::Library> library = Own(device.newLibrary(source, options.get(), &error));
            if (!library)
            {
                fprintf(stderr, "[rhi/metal] shader didn't compile: %s\n", Describe(error).c_str());
                return {};
            }
            return Own(library->newFunction(NS::String::string(entry.c_str(), NS::UTF8StringEncoding)));
        }

    }
}
