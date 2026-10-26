// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
// SPIR-V to MSL (SPIRV-Cross) and MSL to a Metal function; see metal_device.cpp.
#include "metal_internal.h"

#include <spirv_cross/spirv_msl.hpp>

namespace gpu::rhi::metal
{
    namespace impl
    {
        // MSL for `code` as stage `model` against `layout`; the entry point's
        // name and (compute) workgroup size come back in `entry`, `size`.
        std::string ToMsl(const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, std::string& entry, MTL::Size* size, bool pointSize)
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

        // A function of `msl` named `entry`, or an empty pointer (logged).
        Ptr<MTL::Function> MakeFunction(MTL::Device& device, const std::string& msl, const std::string& entry)
        {
            Pool pool;
            auto options = Own(MTL::CompileOptions::alloc()->init());
            options->setLanguageVersion(MTL::LanguageVersion3_0);
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
