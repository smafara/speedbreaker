// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include "vulkan_device.h"
#include "vulkan_internal.h"
#include "vulkan_native.h"

namespace gpu::rhi::vulkan
{
    using namespace impl;

    namespace
    {
        class VkDeviceObject final : public Device
        {
        public:
            explicit VkDeviceObject(std::unique_ptr<Context> context) : m_c(std::move(context)) {}
            ~VkDeviceObject() override
            {
                vkDeviceWaitIdle(m_c->device);
                m_c->imageMemory.reset();
                vkDestroyPipelineCache(m_c->device, m_c->pipelineCache, nullptr);
                vkDestroyDevice(m_c->device, nullptr);
                vkDestroyInstance(m_c->instance, nullptr);
            }

            const Capabilities& GetCapabilities() const override { return m_c->caps; }
            Context* Ctx() { return m_c.get(); }
            const vulkan::Native& Native() const { return m_c->native; }

            std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override
            {
                if (desc.memory == MemoryKind::Stream)
                    return CreateStreamBuffer(desc);
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = desc.size;
                bci.usage = ToVk(BufferUsage(desc.usage));
                VkBuffer buffer;
                if (vkCreateBuffer(m_c->device, &bci, nullptr, &buffer) != VK_SUCCESS)
                    return nullptr;
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                VkMemoryPropertyFlags want = desc.memory == MemoryKind::Device
                    ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    : desc.memory == MemoryKind::Upload
                        ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        : VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                uint32_t type = m_c->FindMemoryType(req.memoryTypeBits, want);
                // Cached where there is some (the CPU reads it), else any coherent host-visible type.
                if (type == UINT32_MAX && desc.memory == MemoryKind::UploadCached)
                    type = m_c->FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (type != UINT32_MAX)
                {
                    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                    mai.allocationSize = req.size;
                    mai.memoryTypeIndex = type;
                    if (vkAllocateMemory(m_c->device, &mai, nullptr, &memory) != VK_SUCCESS)
                        memory = VK_NULL_HANDLE;
                }
                if (!memory || vkBindBufferMemory(m_c->device, buffer, memory, 0) != VK_SUCCESS)
                {
                    vkDestroyBuffer(m_c->device, buffer, nullptr);
                    if (memory)
                        vkFreeMemory(m_c->device, memory, nullptr);
                    return nullptr;
                }
                void* mapped = nullptr;
                if (desc.memory != MemoryKind::Device)
                    Check(vkMapMemory(m_c->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
                bool coherent = (m_c->memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                return std::make_unique<VkBufferObject>(*m_c, buffer, memory, desc.size, mapped, coherent);
            }

            std::unique_ptr<Buffer> ImportHostMemory(void* host, uint64_t size, uint32_t usage, const char*) override
            {
                if (!m_c->hostPointerProperties)
                    return nullptr;
                const uint64_t align = m_c->caps.hostMemoryAlignment;
                if (reinterpret_cast<uintptr_t>(host) % align || size % align)
                {
                    fprintf(stderr, "[rhi/vulkan] host memory import: %p + %llu isn't aligned to %llu\n", host, (unsigned long long)size,
                        (unsigned long long)align);
                    return nullptr;
                }
                VkMemoryHostPointerPropertiesEXT hpp{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
                if (m_c->hostPointerProperties(m_c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp) != VK_SUCCESS)
                    return nullptr;
                VkExternalMemoryBufferCreateInfo embci{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
                embci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.pNext = &embci;
                bci.size = size;
                bci.usage = ToVk(BufferUsage(usage));
                VkBuffer buffer;
                if (vkCreateBuffer(m_c->device, &bci, nullptr, &buffer) != VK_SUCCESS)
                    return nullptr;
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                VkImportMemoryHostPointerInfoEXT import{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
                import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                import.pHostPointer = host;
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.pNext = &import;
                mai.allocationSize = size;
                mai.memoryTypeIndex = m_c->FindMemoryType(req.memoryTypeBits & hpp.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(m_c->device, &mai, nullptr, &memory) != VK_SUCCESS ||
                    vkBindBufferMemory(m_c->device, buffer, memory, 0) != VK_SUCCESS)
                {
                    vkDestroyBuffer(m_c->device, buffer, nullptr);
                    if (memory)
                        vkFreeMemory(m_c->device, memory, nullptr);
                    return nullptr;
                }
                auto result = std::make_unique<VkBufferObject>(*m_c, buffer, memory, size, host, true);
                result->MarkImported();
                return result;
            }


            // ---- where the big CPU-written allocations go (gpu/memory_plan.h, D14) ----
            //
            // Guest memory and the upload ring: a plan lists the places to try
            // in turn; a place the driver refuses (out of memory, say) is logged
            // and the next one tried; only when none is left does the caller
            // give up. Test hooks for the plan's fallbacks, which the machines
            // we test on never need:
            //   NFSMW_TEST_BAR_HEAP_MB=<MB>: the heaps holding CPU-mapped VRAM
            //     read as that size to the plan, and the copy's and the ring's
            //     allocations from those types fail (out of device memory) once
            //     they would pass it, as on a GPU without resizable BAR;
            //   NFSMW_TEST_ALLOC_FAIL=<places>: those places fail as out of
            //     device memory (comma-separated: import, copy-vram,
            //     copy-system, copy-cached, ring-vram, ring-cached, ring-system);
            //   NFSMW_TEST_GPU_KIND=integrated|discrete|other: the plan takes
            //     the GPU for that kind (the Deck's choice on the Steam Machine).
            struct MemoryTest
            {
                uint64_t barBytes = 0, barUsed = 0;
                std::string fail;  // ",a,b,"
                const char* kind = nullptr;
            };
            MemoryTest m_test = [] {
                MemoryTest t;
                if (const char* v = std::getenv("NFSMW_TEST_BAR_HEAP_MB"); v && std::atoi(v) > 0)
                    t.barBytes = uint64_t(std::atoi(v)) << 20;
                if (const char* v = std::getenv("NFSMW_TEST_ALLOC_FAIL"); v && v[0])
                    t.fail = std::string(",") + v + ",";
                t.kind = std::getenv("NFSMW_TEST_GPU_KIND");
                return t;
            }();

            bool MemoryTestRefuses(memplan::Place place, bool ring)
            {
                const char* name = memplan::Name(place, ring);
                if (m_test.fail.find(std::string(",") + name + ",") == std::string::npos)
                    return false;
                fprintf(stderr, "[rhi/vulkan] NFSMW_TEST_ALLOC_FAIL: %s refused (out of device memory)\n", name);
                return true;
            }

            // The first type among `bits` with `want`, else the first among
            // `bits` (the memory is the driver's to say yes or no to).
            uint32_t FindTypeLoose(uint32_t bits, VkMemoryPropertyFlags want) const
            {
                const uint32_t type = m_c->FindMemoryType(bits, want);
                if (type != UINT32_MAX)
                    return type;
                for (uint32_t i = 0; i < m_c->memory.memoryTypeCount; i++)
                    if (bits & (1u << i))
                        return i;
                return 0;
            }

            // vkAllocateMemory for a planned place, through the test hooks.
            VkResult AllocatePlanned(const memplan::Candidate& c, bool ring, VkDeviceSize size, VkDeviceMemory& memory)
            {
                if (MemoryTestRefuses(c.place, ring))
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                const VkMemoryPropertyFlags bar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                const bool mappedVram = (m_c->memory.memoryTypes[c.type].propertyFlags & bar) == bar;
                if (mappedVram && m_test.barBytes && m_test.barUsed + size > m_test.barBytes)
                {
                    fprintf(stderr, "[rhi/vulkan] NFSMW_TEST_BAR_HEAP_MB: %llu MB of CPU-mapped VRAM refused (%llu of %llu MB taken)\n",
                        (unsigned long long)(size >> 20), (unsigned long long)(m_test.barUsed >> 20), (unsigned long long)(m_test.barBytes >> 20));
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.allocationSize = size;
                mai.memoryTypeIndex = c.type;
                const VkResult r = vkAllocateMemory(m_c->device, &mai, nullptr, &memory);
                if (r == VK_SUCCESS && mappedVram)
                    m_test.barUsed += size;
                return r;
            }

            // The device's memory as the plan sees it (NFSMW_TEST_BAR_HEAP_MB
            // applied), logged once: bug reports then show what a GPU offers.
            void PlanDeviceMemory()
            {
                if (m_c->planned)
                    return;
                m_c->planned = true;
                const VkPhysicalDeviceMemoryProperties& mp = m_c->memory;
                memplan::Memory& m = m_c->planMemory;
                std::string line;
                for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
                {
                    m.heaps.push_back(mp.memoryHeaps[h].size);
                    line += std::format("{}{} {}{}", h ? ", " : "", h, memplan::Size(m.heaps.back()),
                        (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " device-local" : "");
                }
                line += "; types (type:heap/flags):";
                for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                {
                    m.types.push_back({ mp.memoryTypes[i].propertyFlags, mp.memoryTypes[i].heapIndex });
                    line += std::format(" {}:{}/{:X}", i, m.types.back().heap, m.types.back().flags);
                }
                if (m_test.barBytes)
                {
                    memplan::ShrinkMappedVram(m, m_test.barBytes);
                    line += std::format(" (NFSMW_TEST_BAR_HEAP_MB: CPU-mapped VRAM's heaps read as {} MB)", m_test.barBytes >> 20);
                }
                fprintf(stderr, "[rhi/vulkan] memory heaps: %s\n", line.c_str());
            }

            uint32_t DriverId() const
            {
                if (m_c->props.apiVersion < VK_API_VERSION_1_2)
                    return 0;
                VkPhysicalDeviceDriverProperties driver{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
                VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver };
                vkGetPhysicalDeviceProperties2(m_c->physical, &props2);
                return uint32_t(driver.driverID);
            }

            // Guest physical memory imported without a copy (unified memory, or
            // a discrete GPU reading system memory over PCIe).
            std::unique_ptr<Buffer> ImportGuest(void* host, uint64_t size)
            {
                if (!m_c->hostPointerProperties)  // (no host-memory import)
                    return nullptr;
                if (MemoryTestRefuses(memplan::Place::kImport, false))
                {
                    fprintf(stderr, "[rhi/vulkan] importing guest memory failed (%d)\n", int(VK_ERROR_OUT_OF_DEVICE_MEMORY));
                    return nullptr;
                }
                VkMemoryHostPointerPropertiesEXT hpp{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
                if (VkResult r = m_c->hostPointerProperties(m_c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp);
                    r != VK_SUCCESS)
                {
                    // (A driver can refuse the pointer itself: on Windows guest
                    // memory is a view of a file mapping, which only Intel's
                    // driver is known to import. Logged, so a bug report tells
                    // this from a refused allocation.)
                    fprintf(stderr, "[rhi/vulkan] importing guest memory failed: the driver doesn't take its host pointer (%d)\n", int(r));
                    return nullptr;
                }
                VkExternalMemoryBufferCreateInfo embci{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
                embci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.pNext = &embci;
                bci.size = size;
                bci.usage = ToVk(BufferUsage(kBufferStorage | kBufferCopySrc | kBufferCopyDst));
                VkBuffer buffer = VK_NULL_HANDLE;
                Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(shared)");
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                VkImportMemoryHostPointerInfoEXT import{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
                import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                import.pHostPointer = host;
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.pNext = &import;
                mai.allocationSize = size;
                mai.memoryTypeIndex = FindTypeLoose(req.memoryTypeBits & hpp.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                VkResult r = vkAllocateMemory(m_c->device, &mai, nullptr, &memory);
                if (r == VK_SUCCESS && (r = vkBindBufferMemory(m_c->device, buffer, memory, 0)) != VK_SUCCESS)
                {
                    vkFreeMemory(m_c->device, memory, nullptr);
                    memory = VK_NULL_HANDLE;
                }
                if (r != VK_SUCCESS)
                {
                    fprintf(stderr, "[rhi/vulkan] importing guest memory failed (%d)\n", int(r));
                    vkDestroyBuffer(m_c->device, buffer, nullptr);
                    return nullptr;
                }
                auto result = std::make_unique<VkBufferObject>(*m_c, buffer, memory, size, host, true);
                result->MarkImported();
                return result;
            }

            // The buffer of guest memory's copy (an unbound one: a place that
            // fails before binding leaves it for the next).
            VkBuffer NewCopyBuffer(uint64_t size, VkMemoryRequirements& req)
            {
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = size;
                bci.usage = ToVk(BufferUsage(kBufferStorage | kBufferCopySrc | kBufferCopyDst));
                VkBuffer buffer = VK_NULL_HANDLE;
                Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(shadow)");
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                return buffer;
            }

            // Guest memory's copy for the GPU in `c`'s memory type: on a
            // discrete GPU in VRAM the CPU writes through resizable BAR when
            // it holds the copy, else in system memory; on an integrated one
            // in system memory (memory_plan.h). False, with nothing kept, when
            // the driver refuses it; `buffer` is then still unbound, or replaced.
            std::unique_ptr<Buffer> CreateShadowGuest(uint64_t size, VkBuffer& buffer, const VkMemoryRequirements& req,
                const memplan::Candidate& c, void*& mapped)
            {
                VkDeviceMemory memory = VK_NULL_HANDLE;
                void* p = nullptr;
                const char* what = "vkAllocateMemory(shadow)";
                VkResult r = AllocatePlanned(c, false, req.size, memory);
                if (r == VK_SUCCESS && (r = vkMapMemory(m_c->device, memory, 0, VK_WHOLE_SIZE, 0, &p)) != VK_SUCCESS)
                    what = "vkMapMemory(shadow)";
                else if (r == VK_SUCCESS && (r = vkBindBufferMemory(m_c->device, buffer, memory, 0)) != VK_SUCCESS)
                {
                    what = "vkBindBufferMemory(shadow)";
                    vkDestroyBuffer(m_c->device, buffer, nullptr);  // (a buffer binds once)
                    VkMemoryRequirements again;
                    buffer = NewCopyBuffer(size, again);
                }
                if (r != VK_SUCCESS)
                {
                    fprintf(stderr, "[rhi/vulkan] %s failed: %d (%s, memory type %u): trying the next place\n", what, int(r),
                        memplan::Name(c.place, false), c.type);
                    if (memory)
                        vkFreeMemory(m_c->device, memory, nullptr);
                    return nullptr;
                }
                mapped = p;
                const bool coherent = (m_c->memory.memoryTypes[c.type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                return std::make_unique<VkBufferObject>(*m_c, buffer, memory, size, p, coherent);
            }

            // `failed`, a list of place names, as the log's ", after ... failed".
            static std::string AfterFailed(const std::string& failed)
            {
                return failed.empty() ? std::string() : " after " + failed + " failed";
            }

            bool CreateGuestMemory(void* host, uint64_t size, GuestMemory& out) override
            {
                const VkPhysicalDeviceMemoryProperties& mp = m_c->memory;
                PlanDeviceMemory();
                memplan::GuestInput in;
                in.gpu = m_c->caps.kind == Capabilities::Kind::Discrete ? memplan::Gpu::kDiscrete
                    : m_c->caps.kind == Capabilities::Kind::Integrated ? memplan::Gpu::kIntegrated : memplan::Gpu::kOther;
                if (const char* kind = m_test.kind)
                {
                    std::string_view k(kind);
                    in.gpu = k == "integrated" ? memplan::Gpu::kIntegrated : k == "discrete" ? memplan::Gpu::kDiscrete
                        : k == "other" ? memplan::Gpu::kOther : in.gpu;
                    fprintf(stderr, "[rhi/vulkan] NFSMW_TEST_GPU_KIND=%s: memory planned as for %s GPU\n", kind,
                        in.gpu == memplan::Gpu::kDiscrete ? "a discrete" : in.gpu == memplan::Gpu::kIntegrated ? "an integrated" : "another");
                }
                in.importable = m_c->hostPointerProperties != nullptr;
                in.importRefusal = memplan::ImportRefusal(DriverId());
                in.overrides = memplan::ParseOverrides(std::getenv("NFSMW_SHARED_MEMORY"), std::getenv("NFSMW_SHADOW_MEMORY"),
                    std::getenv("NFSMW_RING_MEMORY"));
                VkMemoryRequirements req;
                VkBuffer copy = NewCopyBuffer(size, req);
                in.bits = req.memoryTypeBits;
                m_c->planGpu = in.gpu;
                memplan::Plan plan = memplan::PlanGuestMemory(m_c->planMemory, in);
                std::string failed;
                const memplan::Candidate* placed = nullptr;
                for (const memplan::Candidate& c : plan.order)
                {
                    void* mapped = nullptr;
                    if (c.place == memplan::Place::kImport)
                        out.buffer = ImportGuest(host, size);
                    else
                        out.buffer = CreateShadowGuest(size, copy, req, c, mapped);
                    if (out.buffer)
                    {
                        placed = &c;
                        out.shadow = static_cast<uint8_t*>(mapped);
                        break;
                    }
                    failed += std::string(failed.empty() ? "" : ", ") + memplan::Name(c.place, false);
                }
                if (!placed)
                {
                    fprintf(stderr, "[rhi/vulkan] guest memory: no place could hold it (%s failed; %s)\n", failed.c_str(), plan.why.c_str());
                    vkDestroyBuffer(m_c->device, copy, nullptr);
                    return false;
                }
                if (placed->place == memplan::Place::kImport)
                {
                    vkDestroyBuffer(m_c->device, copy, nullptr);
                    fprintf(stderr, "[rhi/vulkan] guest memory: imported (zero-copy)\n");
                    m_c->guestChoice = "imported (zero-copy)";
                }
                else
                {
                    m_c->guestType = placed->type;
                    m_c->shadowInVram = (mp.memoryTypes[placed->type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                    out.shadowInVram = m_c->shadowInVram;
                    fprintf(stderr, "[rhi/vulkan] guest memory: GPU copy in %s (memory type %u, flags %X), updated from the write watch\n",
                        m_c->shadowInVram ? "VRAM (CPU-mapped)" : "host memory", placed->type, mp.memoryTypes[placed->type].propertyFlags);
                    m_c->guestChoice = std::format("as a GPU copy in {}, memory type {}", placed->place == memplan::Place::kVram ? "CPU-mapped VRAM"
                        : placed->place == memplan::Place::kCached ? "cached host memory" : "host memory", placed->type);
                }
                m_c->guestChoice += " (" + plan.why + ")" + AfterFailed(failed);
                return true;
            }

            // The upload ring: device-local when the host can map it
            // (resizable BAR) and it fits, so the GPU reads constants and
            // indices from VRAM; else cached system memory (memory_plan.h).
            // NFSMW_RING_MEMORY=cached prefers cached system memory anyway: the
            // CPU's ~40 MB of writes a frame then aren't write-combined PCIe
            // stores (an A/B experiment: on the Steam Machine 2% less CPU,
            // twice the GPU time); =vram|system name the other places.
            std::unique_ptr<Buffer> CreateStreamBuffer(const BufferDesc& desc)
            {
                PlanDeviceMemory();
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = desc.size;
                bci.usage = ToVk(BufferUsage(desc.usage));
                VkBuffer buffer = VK_NULL_HANDLE;
                Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(ring)");
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                const memplan::Overrides o = memplan::ParseOverrides(nullptr, nullptr, std::getenv("NFSMW_RING_MEMORY"));
                memplan::Plan plan = memplan::PlanRing(m_c->planMemory, req.memoryTypeBits, m_c->guestType, o.ring);
                std::string failed;
                for (const memplan::Candidate& c : plan.order)
                {
                    VkDeviceMemory memory = VK_NULL_HANDLE;
                    void* p = nullptr;
                    const char* what = "vkAllocateMemory(ring)";
                    VkResult r = AllocatePlanned(c, true, req.size, memory);
                    if (r == VK_SUCCESS && (r = vkMapMemory(m_c->device, memory, 0, VK_WHOLE_SIZE, 0, &p)) != VK_SUCCESS)
                        what = "vkMapMemory(ring)";
                    else if (r == VK_SUCCESS && (r = vkBindBufferMemory(m_c->device, buffer, memory, 0)) != VK_SUCCESS)
                    {
                        what = "vkBindBufferMemory(ring)";
                        vkDestroyBuffer(m_c->device, buffer, nullptr);  // (a buffer binds once)
                        Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(ring)");
                    }
                    if (r != VK_SUCCESS)
                    {
                        fprintf(stderr, "[rhi/vulkan] %s failed: %d (%s, memory type %u): trying the next place\n", what, int(r),
                            memplan::Name(c.place, true), c.type);
                        if (memory)
                            vkFreeMemory(m_c->device, memory, nullptr);
                        failed += std::string(failed.empty() ? "" : ", ") + memplan::Name(c.place, true);
                        continue;
                    }
                    const VkMemoryPropertyFlags flags = m_c->memory.memoryTypes[c.type].propertyFlags;
                    fprintf(stderr, "[rhi/vulkan] upload ring: memory type %u (flags %X)\n", c.type, flags);
                    // The choice and why, in one line.
                    fprintf(stderr, "[rhi/vulkan] memory: guest memory %s; upload ring in %s, memory type %u (%s)%s\n", m_c->guestChoice.c_str(),
                        c.place == memplan::Place::kVram ? "CPU-mapped VRAM" : c.place == memplan::Place::kCached ? "cached host memory"
                            : "host memory", c.type, plan.why.c_str(), AfterFailed(failed).c_str());
                    // Issue #40: say so when the GPU reads guest memory across PCIe.
                    if (const char* note = memplan::OverPcieNote(m_c->planGpu, m_c->guestType != memplan::kNoType && m_c->shadowInVram,
                            (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0))
                        fprintf(stderr, "[rhi/vulkan] note: %s\n", note);
                    return std::make_unique<VkBufferObject>(*m_c, buffer, memory, desc.size, p,
                        (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0);
                }
                fprintf(stderr, "[rhi/vulkan] upload ring: no place could hold it (%s failed; %s)\n", failed.c_str(), plan.why.c_str());
                vkDestroyBuffer(m_c->device, buffer, nullptr);
                return nullptr;
            }

            std::unique_ptr<Image> CreateImage(const ImageDesc& desc) override
            {
                const auto start = std::chrono::steady_clock::now();
                const VkFormat format = ToVk(desc.format);
                const bool cube = desc.kind == ImageKind::Cube;
                const bool store = desc.storeFormat != Format::Undefined;
                // A store format: views in that format of an image whose own
                // isn't storable, with the usage the view says (the image is
                // sampled through views of its own format).
                const VkFormat formats[2] = { format, store ? ToVk(desc.storeFormat) : format };
                VkImageFormatListCreateInfo list{ VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, nullptr, 2, formats };
                VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
                ici.pNext = store ? &list : nullptr;
                ici.flags = (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
                    (store ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0);
                ici.imageType = desc.kind == ImageKind::Image3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
                ici.format = format;
                ici.extent = { desc.extent.width, desc.extent.height, desc.extent.depth };
                ici.mipLevels = desc.levels;
                ici.arrayLayers = cube ? 6 : desc.layers;
                ici.samples = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling = VK_IMAGE_TILING_OPTIMAL;  // (image memory blocks hold nothing linear)
                ici.usage = ToVk(ImageUsage(desc.usage));
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VkImage image;
                if (vkCreateImage(m_c->device, &ici, nullptr, &image) != VK_SUCCESS)
                    return nullptr;
                ImageMemory::Allocation memory;
                if (!m_c->imageMemory->Bind(image, memory))
                {
                    fprintf(stderr, "[rhi/vulkan] no memory for an image (%ux%u, format %d)\n", desc.extent.width, desc.extent.height, int(format));
                    vkDestroyImage(m_c->device, image, nullptr);
                    return nullptr;
                }
                VkImageViewUsageCreateInfo sampled{ VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, VK_IMAGE_USAGE_SAMPLED_BIT };
                VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                vci.pNext = store ? &sampled : nullptr;
                vci.image = image;
                vci.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : desc.kind == ImageKind::Image3D ? VK_IMAGE_VIEW_TYPE_3D
                    : desc.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                vci.format = format;
                vci.subresourceRange = { Aspect(desc.format), 0, desc.levels, 0, cube ? 6 : desc.layers };
                VkImageView view;
                if (vkCreateImageView(m_c->device, &vci, nullptr, &view) != VK_SUCCESS)
                {
                    vkDestroyImage(m_c->device, image, nullptr);
                    m_c->imageMemory->Free(memory);
                    return nullptr;
                }
                auto result = std::make_unique<VkImageObject>(*m_c, desc, image, memory.memory, view, memory.bytes);
                result->SetAllocation(memory);
                m_c->imageMemory->Made(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
                return result;
            }

            std::unique_ptr<ImageView> CreateImageView(const Image& image, const ImageViewDesc& desc) override
            {
                const auto& img = static_cast<const VkImageObject&>(image);
                const ImageDesc& d = img.Desc();
                auto component = [](Swizzle s) {
                    switch (s)
                    {
                    case Swizzle::R: return VK_COMPONENT_SWIZZLE_R;
                    case Swizzle::G: return VK_COMPONENT_SWIZZLE_G;
                    case Swizzle::B: return VK_COMPONENT_SWIZZLE_B;
                    case Swizzle::A: return VK_COMPONENT_SWIZZLE_A;
                    case Swizzle::Zero: return VK_COMPONENT_SWIZZLE_ZERO;
                    case Swizzle::One: return VK_COMPONENT_SWIZZLE_ONE;
                    default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                    }
                };
                const Format format = desc.format == Format::Undefined ? d.format : desc.format;
                const bool cube = d.kind == ImageKind::Cube;
                VkImageViewUsageCreateInfo usage{ VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, 0 };
                VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                if (format != d.format)  // (a reinterpretation: the view's own usage is storage)
                {
                    usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
                    vci.pNext = &usage;
                }
                vci.image = img.Handle();
                vci.format = ToVk(format);
                if (desc.layerCount)
                {
                    vci.viewType = desc.layerCount > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                    vci.subresourceRange = { Aspect(format), 0, 1, desc.layer, desc.layerCount };
                }
                else
                {
                    vci.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : d.kind == ImageKind::Image3D ? VK_IMAGE_VIEW_TYPE_3D
                        : d.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                    vci.subresourceRange = { Aspect(format), 0, d.levels, 0, cube ? 6 : d.layers };
                }
                if (desc.depthOnly)
                    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                vci.components = { component(desc.swizzle[0]), component(desc.swizzle[1]), component(desc.swizzle[2]), component(desc.swizzle[3]) };
                VkImageView view;
                if (vkCreateImageView(m_c->device, &vci, nullptr, &view) != VK_SUCCESS)
                    return nullptr;
                return std::make_unique<VkImageViewObject>(*m_c, view);
            }

            bool SupportsImage(const ImageDesc& desc) const override
            {
                const bool cube = desc.kind == ImageKind::Cube;
                const bool store = desc.storeFormat != Format::Undefined;
                // (The image's format list is Vulkan 1.2.)
                if (store && m_c->props.apiVersion < VK_API_VERSION_1_2)
                    return false;
                if (store)
                {
                    VkFormatProperties fp;
                    vkGetPhysicalDeviceFormatProperties(m_c->physical, ToVk(desc.storeFormat), &fp);
                    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
                        return false;
                }
                VkImageFormatProperties props;
                if (vkGetPhysicalDeviceImageFormatProperties(m_c->physical, ToVk(desc.format), VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                        ToVk(ImageUsage(desc.usage)),
                        (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
                            (store ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0),
                        &props) != VK_SUCCESS)
                    return false;
                return !cube || props.maxArrayLayers >= 6;
            }

            uint32_t FormatFeatures(Format format) const override
            {
                VkFormatProperties fp;
                vkGetPhysicalDeviceFormatProperties(m_c->physical, ToVk(format), &fp);
                return ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) ? kFormatBlend : 0) |
                    ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? kFormatStorage : 0) |
                    ((fp.optimalTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ==
                            (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
                        ? kFormatSampleLinear : 0);
            }

            void GetMemoryUsage(uint64_t& used, uint64_t& budget) const override
            {
                used = budget = 0;
                if (!m_c->memoryBudget)
                    return;
                VkPhysicalDeviceMemoryBudgetPropertiesEXT info{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
                VkPhysicalDeviceMemoryProperties2 props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &info };
                vkGetPhysicalDeviceMemoryProperties2(m_c->physical, &props);
                for (uint32_t h = 0; h < props.memoryProperties.memoryHeapCount; h++)
                    if (props.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    {
                        used += info.heapUsage[h];
                        budget += info.heapBudget[h];
                    }
            }

            ImageMemoryStats GetImageMemoryStats() const override { return m_c->imageMemory->Stats(); }

            std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override
            {
                auto filter = [](Filter f) { return f == Filter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; };
                auto address = [](AddressMode a) {
                    return a == AddressMode::ClampToEdge ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                        : a == AddressMode::ClampToBorder ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
                        : a == AddressMode::MirrorRepeat ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT : VK_SAMPLER_ADDRESS_MODE_REPEAT;
                };
                VkSamplerCreateInfo ci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
                ci.magFilter = filter(desc.mag);
                ci.minFilter = filter(desc.min);
                ci.mipmapMode = desc.mip == Filter::Nearest ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
                ci.addressModeU = address(desc.u);
                ci.addressModeV = address(desc.v);
                ci.addressModeW = address(desc.w);
                ci.anisotropyEnable = desc.maxAnisotropy > 1.0f ? VK_TRUE : VK_FALSE;
                ci.maxAnisotropy = desc.maxAnisotropy;
                ci.minLod = desc.minLod;
                ci.maxLod = desc.maxLod >= 1000.0f ? VK_LOD_CLAMP_NONE : desc.maxLod;
                ci.borderColor = desc.whiteBorder ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                VkSampler sampler;
                Check(vkCreateSampler(m_c->device, &ci, nullptr, &sampler), "vkCreateSampler");
                return std::make_unique<VkSamplerObject>(*m_c, sampler);
            }

            std::unique_ptr<BindingLayout> CreateBindingLayout(const BindingLayoutDesc& desc) override
            {
                return std::make_unique<VkBindingLayout>(*m_c, desc);
            }
            std::unique_ptr<BindingSet> CreateBindingSet(const BindingLayout& layout, uint32_t table) override
            {
                return std::make_unique<VkBindingSet>(*m_c, static_cast<const VkBindingLayout&>(layout), table);
            }

            std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) override
            {
                VkShaderModule modules[2] = {};
                const ShaderCode* codes[2] = { &desc.vertex, &desc.fragment };
                VkPipelineShaderStageCreateInfo stages[2]{};
                for (int i = 0; i < 2; i++)
                {
                    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                    smci.codeSize = codes[i]->words * 4;
                    smci.pCode = codes[i]->spirv;
                    if (vkCreateShaderModule(m_c->device, &smci, nullptr, &modules[i]) != VK_SUCCESS)
                    {
                        for (VkShaderModule m : modules)
                            if (m)
                                vkDestroyShaderModule(m_c->device, m, nullptr);
                        return nullptr;
                    }
                    stages[i] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                        i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT, modules[i], "main", nullptr };
                }
                VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };  // (vertices come from buffers in the shader)
                VkPipelineInputAssemblyStateCreateInfo assembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
                assembly.topology = ToVk(desc.topology);
                assembly.primitiveRestartEnable = desc.primitiveRestart ? VK_TRUE : VK_FALSE;
                VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
                viewport.viewportCount = viewport.scissorCount = 1;
                VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
                raster.depthClampEnable = desc.depthClamp ? VK_TRUE : VK_FALSE;
                raster.polygonMode = VK_POLYGON_MODE_FILL;
                raster.cullMode = desc.cull == CullMode::Front ? VK_CULL_MODE_FRONT_BIT : desc.cull == CullMode::Back ? VK_CULL_MODE_BACK_BIT
                                                                                                                       : VK_CULL_MODE_NONE;
                raster.frontFace = desc.frontCounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
                raster.depthBiasEnable = desc.depthBias ? VK_TRUE : VK_FALSE;
                raster.lineWidth = 1.0f;
                VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
                multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
                depth.depthTestEnable = desc.depthTest ? VK_TRUE : VK_FALSE;
                depth.depthWriteEnable = desc.depthWrite ? VK_TRUE : VK_FALSE;
                depth.depthCompareOp = ToVk(desc.depthCompare);
                depth.stencilTestEnable = desc.stencilTest ? VK_TRUE : VK_FALSE;
                auto face = [](const StencilFace& f) {
                    VkStencilOpState s{};
                    s.failOp = ToVk(f.fail);
                    s.passOp = ToVk(f.pass);
                    s.depthFailOp = ToVk(f.depthFail);
                    s.compareOp = ToVk(f.compare);
                    return s;
                };
                depth.front = face(desc.stencilFront);
                depth.back = face(desc.stencilBack);
                VkPipelineColorBlendAttachmentState attachments[4]{};
                VkFormat colorFormats[4] = {};
                constexpr uint32_t colorCount = 4;  // (see BeginRendering; a gap has no format and writes nothing)
                for (uint32_t i = 0; i < 4; i++)
                {
                    colorFormats[i] = ToVk(desc.colorFormats[i]);
                    if (desc.colorFormats[i] == Format::Undefined)
                        continue;
                    const BlendAttachment& b = desc.blend[i];
                    attachments[i].blendEnable = b.enable ? VK_TRUE : VK_FALSE;
                    attachments[i].srcColorBlendFactor = ToVk(b.srcColor);
                    attachments[i].dstColorBlendFactor = ToVk(b.dstColor);
                    attachments[i].colorBlendOp = ToVk(b.colorOp);
                    attachments[i].srcAlphaBlendFactor = ToVk(b.srcAlpha);
                    attachments[i].dstAlphaBlendFactor = ToVk(b.dstAlpha);
                    attachments[i].alphaBlendOp = ToVk(b.alphaOp);
                    attachments[i].colorWriteMask = b.writeMask;  // (R, G, B, A are the low four bits in both)
                }
                VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
                blend.attachmentCount = colorCount;
                blend.pAttachments = attachments;
                const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_DEPTH_BIAS,
                    VK_DYNAMIC_STATE_BLEND_CONSTANTS };
                VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
                dynamicState.dynamicStateCount = uint32_t(sizeof(dynamic) / sizeof(dynamic[0]));
                dynamicState.pDynamicStates = dynamic;
                VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
                rendering.colorAttachmentCount = colorCount;
                rendering.pColorAttachmentFormats = colorFormats;
                if (desc.depthFormat != Format::Undefined)
                {
                    rendering.depthAttachmentFormat = ToVk(desc.depthFormat);
                    rendering.stencilAttachmentFormat = HasDepth(desc.depthFormat) ? ToVk(desc.depthFormat) : VK_FORMAT_UNDEFINED;
                }
                // An experiment (NFSMW_SHADING_RATE): every draw shades once per w x h pixels.
                VkPipelineFragmentShadingRateStateCreateInfoKHR rate{ VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR };
                if (desc.shadingRateW)
                {
                    rate.fragmentSize = { desc.shadingRateW, desc.shadingRateH };
                    rate.combinerOps[0] = rate.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
                    rendering.pNext = &rate;
                }
                VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
                gpci.pNext = &rendering;
                gpci.stageCount = 2;
                gpci.pStages = stages;
                gpci.pVertexInputState = &vertexInput;
                gpci.pInputAssemblyState = &assembly;
                gpci.pViewportState = &viewport;
                gpci.pRasterizationState = &raster;
                gpci.pMultisampleState = &multisample;
                gpci.pDepthStencilState = &depth;
                gpci.pColorBlendState = &blend;
                gpci.pDynamicState = &dynamicState;
                gpci.layout = static_cast<const VkBindingLayout*>(desc.layout)->Handle();
                VkPipeline pipeline = VK_NULL_HANDLE;
                VkResult r = vkCreateGraphicsPipelines(m_c->device, m_c->pipelineCache, 1, &gpci, nullptr, &pipeline);
                for (VkShaderModule m : modules)
                    vkDestroyShaderModule(m_c->device, m, nullptr);
                if (r != VK_SUCCESS)
                    return nullptr;
                return std::make_unique<VkPipelineObject>(*m_c, pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS);
            }

            std::unique_ptr<Pipeline> CreateComputePipeline(const ComputePipelineDesc& desc) override
            {
                VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smci.codeSize = desc.code.words * 4;
                smci.pCode = desc.code.spirv;
                VkShaderModule module;
                if (vkCreateShaderModule(m_c->device, &smci, nullptr, &module) != VK_SUCCESS)
                    return nullptr;
                VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                cpci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr };
                cpci.layout = static_cast<const VkBindingLayout*>(desc.layout)->Handle();
                VkPipeline pipeline = VK_NULL_HANDLE;
                VkResult r = vkCreateComputePipelines(m_c->device, m_c->pipelineCache, 1, &cpci, nullptr, &pipeline);
                vkDestroyShaderModule(m_c->device, module, nullptr);
                if (r != VK_SUCCESS)
                    return nullptr;
                return std::make_unique<VkPipelineObject>(*m_c, pipeline, VK_PIPELINE_BIND_POINT_COMPUTE);
            }

            const char* PipelineCacheFile() const override { return "pipelines.bin"; }
            std::vector<uint8_t> SavePipelineCache() override
            {
                size_t size = 0;
                vkGetPipelineCacheData(m_c->device, m_c->pipelineCache, &size, nullptr);
                std::vector<uint8_t> data(size);
                if (size)
                    vkGetPipelineCacheData(m_c->device, m_c->pipelineCache, &size, data.data());
                data.resize(size);
                return data;
            }
            void LoadPipelineCache(const void* data, size_t size) override
            {
                // A cache another GPU, driver or OS version made is set aside
                // by the driver without a word (MoltenVK's device ID carries
                // the macOS version, so every macOS update does it), and the
                // first drive afterwards stutters while the pipelines compile
                // again. The header says so in advance
                // (gpu/pipeline_cache_header.h): one line for the log, and the
                // stale data isn't handed over. The next save writes this
                // device's header, so it's said once.
                gpu::pipeline_cache::Maker device{ m_c->props.vendorID, m_c->props.deviceID, {} };
                std::memcpy(device.uuid.data(), m_c->props.pipelineCacheUUID, device.uuid.size());
                const auto check = gpu::pipeline_cache::CheckHeader(static_cast<const uint8_t*>(data), size, device);
                if (const std::string line = gpu::pipeline_cache::LogLine(check, device); !line.empty())
                {
                    fprintf(stderr, "%s\n", line.c_str());
                    data = nullptr;
                    size = 0;
                }
                VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
                ci.initialDataSize = size;
                ci.pInitialData = data;
                VkPipelineCache cache;
                if (vkCreatePipelineCache(m_c->device, &ci, nullptr, &cache) != VK_SUCCESS)
                    return;  // (the old, empty one stays)
                vkDestroyPipelineCache(m_c->device, m_c->pipelineCache, nullptr);
                m_c->pipelineCache = cache;
            }

            std::unique_ptr<CommandList> CreateCommandList() override { return MakeCommandList(*m_c); }
            std::unique_ptr<Fence> CreateFence(bool signaled) override { return std::make_unique<VkFenceObject>(*m_c, signaled); }
            std::unique_ptr<Event> CreateEvent() override { return std::make_unique<VkEventObject>(*m_c); }
            std::unique_ptr<QueryPool> CreateQueryPool(QueryKind kind, uint32_t count) override
            {
                return std::make_unique<VkQueryPoolObject>(*m_c, kind, count);
            }

            void Submit(CommandList& list, Fence& fence) override
            {
                VkCommandBuffer cmd = CommandBufferOf(list);
                VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmd;
                std::unique_lock lock(m_c->queueMutex);
                if (m_c->submitGate)
                    m_c->submitGate(lock);
                Check(vkQueueSubmit(m_c->queue, 1, &si, static_cast<VkFenceObject&>(fence).Handle()), "vkQueueSubmit");
            }
            void SetSubmitGate(std::function<void(std::unique_lock<std::mutex>&)> gate) override { m_c->submitGate = std::move(gate); }
            std::unique_ptr<Swapchain> CreateSwapchain(Surface& surface, const SwapchainDesc& desc) override
            {
                return MakeSwapchain(*m_c, surface, desc);
            }
            void SubmitToSwapchain(CommandList& list, Fence& fence, Swapchain& swapchain) override
            {
                VkCommandBuffer cmd = CommandBufferOf(list);
                VkSemaphore wait, signal;
                SwapchainSubmitSemaphores(swapchain, wait, signal);
                const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.waitSemaphoreCount = 1;
                si.pWaitSemaphores = &wait;
                si.pWaitDstStageMask = &waitStage;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmd;
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &signal;
                std::unique_lock lock(m_c->queueMutex);
                if (m_c->submitGate)
                    m_c->submitGate(lock);
                Check(vkQueueSubmit(m_c->queue, 1, &si, static_cast<VkFenceObject&>(fence).Handle()), "vkQueueSubmit");
            }
            void WaitIdle() override
            {
                std::lock_guard lock(m_c->queueMutex);
                vkDeviceWaitIdle(m_c->device);
            }

        private:
            std::unique_ptr<Context> m_c;
        };

        bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name)
        {
            return std::any_of(list.begin(), list.end(), [name](const VkExtensionProperties& e) { return strcmp(e.extensionName, name) == 0; });
        }
    }

    std::unique_ptr<Device> CreateHeadlessDevice()
    {
        auto c = std::make_unique<Context>();

        // The instance. On macOS the loader lists MoltenVK only when asked
        // (portability enumeration); iOS and others may not offer it.
        uint32_t count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> instanceExts(count);
        vkEnumerateInstanceExtensionProperties(nullptr, &count, instanceExts.data());
        std::vector<const char*> exts;
        const bool portability = HasExtension(instanceExts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        if (portability)
            exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        if (HasExtension(instanceExts, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "SpeedBreaker";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = uint32_t(exts.size());
        ici.ppEnabledExtensionNames = exts.data();
        if (vkCreateInstance(&ici, nullptr, &c->instance) != VK_SUCCESS)
            return nullptr;

        // The device: a discrete GPU first, with a graphics + compute queue.
        count = 0;
        vkEnumeratePhysicalDevices(c->instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(c->instance, &count, devices.data());
        VkPhysicalDeviceProperties props{};
        for (int pass = 0; pass < 2 && !c->physical; pass++)
            for (VkPhysicalDevice d : devices)
            {
                VkPhysicalDeviceProperties p;
                vkGetPhysicalDeviceProperties(d, &p);
                uint32_t qc = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, nullptr);
                std::vector<VkQueueFamilyProperties> qs(qc);
                vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, qs.data());
                for (uint32_t f = 0; f < qc; f++)
                    if ((qs[f].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) &&
                        (pass == 1 || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU))
                    {
                        c->physical = d;
                        c->family = f;
                        props = p;
                        break;
                    }
                if (c->physical)
                    break;
            }
        if (!c->physical)
        {
            vkDestroyInstance(c->instance, nullptr);
            return nullptr;
        }

        count = 0;
        vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> avail(count);
        vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &count, avail.data());
        std::vector<const char*> devExts;
        const bool pushDescriptor = HasExtension(avail, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        const bool hostImport = HasExtension(avail, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        if (pushDescriptor)
            devExts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        if (hostImport)
            devExts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        if (HasExtension(avail, "VK_KHR_portability_subset"))
            devExts.push_back("VK_KHR_portability_subset");
        // Dynamic rendering, and what MoltenVK needs it to bring along.
        for (const char* name : { VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
                 VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_KHR_MAINTENANCE_2_EXTENSION_NAME })
            if (HasExtension(avail, name))
                devExts.push_back(name);
        VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRendering{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };
        dynamicRendering.dynamicRendering = VK_TRUE;

        VkPhysicalDeviceFeatures supported;
        vkGetPhysicalDeviceFeatures(c->physical, &supported);
        VkPhysicalDeviceFeatures features{};
        features.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;
        features.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
        features.independentBlend = supported.independentBlend;
        features.depthClamp = supported.depthClamp;
        features.samplerAnisotropy = supported.samplerAnisotropy;
        features.textureCompressionBC = supported.textureCompressionBC;
        features.occlusionQueryPrecise = supported.occlusionQueryPrecise;

        DeviceConfig config;
        config.instance = c->instance;
        config.physical = c->physical;
        config.queueFamily = c->family;
        config.extensions = devExts;
        config.features = &features;
        config.next = &dynamicRendering;
        std::unique_ptr<Device> device = CreateDevice(config);
        if (!device)
            vkDestroyInstance(c->instance, nullptr);
        return device;
    }

    std::unique_ptr<Device> CreateDevice(const DeviceConfig& config)
    {
        auto c = std::make_unique<Context>();
        c->instance = config.instance;
        c->physical = config.physical;
        c->family = config.queueFamily;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(c->physical, &props);
        vkGetPhysicalDeviceMemoryProperties(c->physical, &c->memory);
        c->nonCoherentAtom = std::max<VkDeviceSize>(props.limits.nonCoherentAtomSize, 1);
        auto has = [&](const char* name) {
            return std::any_of(config.extensions.begin(), config.extensions.end(), [name](const char* e) { return strcmp(e, name) == 0; });
        };
        const bool pushDescriptor = has(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        const bool hostImport = has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        VkPhysicalDeviceFeatures supported;
        vkGetPhysicalDeviceFeatures(c->physical, &supported);
        c->preciseOcclusion = config.features && config.features->occlusionQueryPrecise;
        bool hostQueryReset = false;
        for (auto* in = static_cast<const VkBaseInStructure*>(config.next); in; in = in->pNext)
            if (in->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES)
                hostQueryReset |= reinterpret_cast<const VkPhysicalDeviceHostQueryResetFeatures*>(in)->hostQueryReset == VK_TRUE;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = c->family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.pNext = config.next;
        dci.pEnabledFeatures = config.features;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = uint32_t(config.extensions.size());
        dci.ppEnabledExtensionNames = config.extensions.data();
        if (VkResult r = vkCreateDevice(c->physical, &dci, nullptr, &c->device); r != VK_SUCCESS)
        {
            fprintf(stderr, "[rhi/vulkan] vkCreateDevice failed (%d)\n", int(r));
            return nullptr;
        }
        vkGetDeviceQueue(c->device, c->family, 0, &c->queue);
        if (hostQueryReset)
            c->hostResetQueries = reinterpret_cast<PFN_vkResetQueryPool>(vkGetDeviceProcAddr(c->device, "vkResetQueryPool"));
        c->native = { c->instance, c->physical, c->device, c->family, c->queue, &c->queueMutex };
        c->pushDescriptor = pushDescriptor ? reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR")) : nullptr;
        c->hostPointerProperties = hostImport ? reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(c->device, "vkGetMemoryHostPointerPropertiesEXT")) : nullptr;
        c->beginRendering = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(vkGetDeviceProcAddr(c->device, "vkCmdBeginRenderingKHR"));
        c->endRendering = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(vkGetDeviceProcAddr(c->device, "vkCmdEndRenderingKHR"));
        // Present timing: the caller enabled these (policy: the settings and
        // the platform); the interface reports what it was given.
        if (has(VK_KHR_PRESENT_ID_EXTENSION_NAME) && has(VK_KHR_PRESENT_WAIT_EXTENSION_NAME))
            c->waitForPresent = reinterpret_cast<PFN_vkWaitForPresentKHR>(vkGetDeviceProcAddr(c->device, "vkWaitForPresentKHR"));
        if (has(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME))
            c->pastTiming = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                vkGetDeviceProcAddr(c->device, "vkGetPastPresentationTimingGOOGLE"));
        VkPipelineCacheCreateInfo pcci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        Check(vkCreatePipelineCache(c->device, &pcci, nullptr, &c->pipelineCache), "vkCreatePipelineCache");

        // What the device offers.
        c->deviceName = props.deviceName;
        c->props = props;
        {
            uint32_t extensionCount = 0;
            vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> extensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &extensionCount, extensions.data());
            for (const VkExtensionProperties& e : extensions)
                c->memoryBudget |= strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
        }
        Context* raw = c.get();
        c->imageMemory = std::make_unique<ImageMemory>(
            c->device, [raw](uint32_t bits) { return raw->FindMemoryType(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); }, config.onThread);
        Capabilities& caps = c->caps;
        caps.kind = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? Capabilities::Kind::Discrete
            : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? Capabilities::Kind::Integrated : Capabilities::Kind::Other;
        caps.vendorId = props.vendorID;
        caps.maxStorageBufferRange = props.limits.maxStorageBufferRange;
        caps.maxAnisotropy = supported.samplerAnisotropy ? props.limits.maxSamplerAnisotropy : 1.0f;
        caps.backend = "Vulkan";
        caps.deviceName = c->deviceName.c_str();
        caps.unifiedMemory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        caps.canImportHostMemory = c->hostPointerProperties != nullptr;
        caps.hostMemoryAlignment = 4096;
        if (caps.canImportHostMemory)
        {
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
            VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &host };
            vkGetPhysicalDeviceProperties2(c->physical, &props2);
            caps.hostMemoryAlignment = std::max<uint64_t>(host.minImportedHostPointerAlignment, 1);
        }
        caps.maxBufferSize = props.limits.maxStorageBufferRange;
        caps.textureCompressionBC = config.features && config.features->textureCompressionBC;
        caps.occlusionQueries = c->preciseOcclusion;
        caps.timestampQueries = props.limits.timestampComputeAndGraphics;
        caps.timestampPeriodNs = props.limits.timestampPeriod;
        caps.presentTiming = c->waitForPresent != nullptr;
#if defined(__APPLE__) && !TARGET_OS_IOS
        caps.scheduledPresent = c->pastTiming != nullptr;
#endif
        return std::make_unique<VkDeviceObject>(std::move(c));
    }

    std::unique_ptr<Surface> CreateSurface(Device& device, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels)
    {
        return MakeSurface(*static_cast<VkDeviceObject&>(device).Ctx(), surface, std::move(windowPixels));
    }

    VkImage NativeImage(const Image& image) { return static_cast<const VkImageObject&>(image).Handle(); }
    VkImageView NativeView(const Image& image) { return static_cast<const VkImageObject&>(image).View(); }
    VkImageView NativeImageView(const ImageView& view) { return static_cast<const VkImageViewObject&>(view).Handle(); }
    VkBuffer NativeBuffer(const Buffer& buffer) { return static_cast<const VkBufferObject&>(buffer).Handle(); }
    VkFormat NativeFormat(const Swapchain& swapchain) { return SwapchainFormat(swapchain); }
    std::unique_ptr<Image> WrapImage(Device& device, VkImage image, Format format, uint32_t width, uint32_t height, uint32_t usage)
    {
        Context& c = *static_cast<VkDeviceObject&>(device).Ctx();
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ToVk(format);
        vci.subresourceRange = { Aspect(format), 0, 1, 0, 1 };
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(c.device, &vci, nullptr, &view) != VK_SUCCESS)
            return nullptr;
        ImageDesc desc;
        desc.format = format;
        desc.extent = { width, height, 1 };
        desc.usage = usage;
        desc.name = "wrapped";
        return std::make_unique<VkImageObject>(c, desc, image, VK_NULL_HANDLE, view, 0);
    }

    VkCommandBuffer NativeCommandBuffer(const CommandList& list) { return CommandBufferOf(list); }

    const Native& GetNative(const rhi::Device& device)
    {
        return static_cast<const VkDeviceObject&>(device).Native();
    }
}

