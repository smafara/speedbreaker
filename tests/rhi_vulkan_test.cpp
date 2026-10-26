// The RHI tests (rhi_test_body.h) on the Vulkan backend.
#include <gpu/rhi/vulkan/vulkan_device.h>

#define RHI_BACKEND_NAME "Vulkan"
static std::unique_ptr<gpu::rhi::Device> MakeDevice() { return gpu::rhi::vulkan::CreateHeadlessDevice(); }

#include "rhi_test_body.h"
