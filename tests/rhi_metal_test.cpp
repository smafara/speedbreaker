// The RHI tests (rhi_test_body.h) on the Metal backend.
#include <gpu/rhi/metal/metal_device.h>

#define RHI_BACKEND_NAME "Metal"
static std::unique_ptr<gpu::rhi::Device> MakeDevice() { return gpu::rhi::metal::CreateDevice(); }

#include "rhi_test_body.h"
