// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// Image memory for the Vulkan backend (D14; it was in gpu/renderer.cpp).
//
// Images are bound at offsets into blocks of device memory (64 MB;
// gpu/block_ranges.h keeps each block's free ranges), one list per memory
// type, instead of an allocation each. On the Deck vkAllocateMemory was most
// of a new texture's cost (40-220 us in scripted runs, 380 us while the
// kernel was moving memory out of VRAM, about 1 ms after a long free roam,
// where frames that streamed textures spent a median 9.5 ms making images);
// a range in a block is a few us. A block is allocated ahead of need, off the
// caller's thread: the nfsmw-memory thread keeps an empty spare of each
// memory type in use (0.1-0.2 ms a block there, unless the kernel has memory
// to move), so the thread that makes images takes the spare when a block
// fills and the memory thread makes the next. Without a spare ready (the
// first block, or a burst that outran the thread) the caller allocates the
// block itself, counted in the stats. An image bigger than half a block, one
// the driver wants in an allocation of its own (VkMemoryDedicatedRequirements)
// and any image when a block can't be had keep their own allocation, as
// before. A block whose images all went is a spare too, freed by the thread
// after a few seconds unused beside another. Images are destroyed only with
// the GPU idle (the renderer's rule, ApplyLiveSettings), so a range can be
// handed out again at once.
// NFSMW_IMAGE_POOL=0: an allocation per image (the old path);
// NFSMW_IMAGE_BLOCK_MB: the block size (16-1024).
//
// Callers: one thread binds and frees (the renderer's command processor);
// the spare thread is this class's own.
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

#include "../../block_ranges.h"
#include "../device.h"

namespace gpu::rhi::vulkan
{
    class ImageMemory
    {
    public:
        // Where an image's memory is: a range of a block, or its own allocation.
        struct Allocation
        {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize offset = 0, bytes = 0;
            void* block = nullptr;   // the block it is a range of, if any
        };

        // `findType`: the memory type for a requirement's bits (device-local,
        // UINT32_MAX if none). `onThread`: called first on the spare thread
        // (its name, for the host to name and place it).
        ImageMemory(VkDevice device, std::function<uint32_t(uint32_t)> findType, std::function<void(const char*)> onThread)
            : m_device(device), m_findType(std::move(findType)), m_onThread(std::move(onThread))
        {
            const char* pool = std::getenv("NFSMW_IMAGE_POOL");
            m_pool = !pool || pool[0] != '0';
            const char* mb = std::getenv("NFSMW_IMAGE_BLOCK_MB");
            m_blockSize = VkDeviceSize(mb ? std::clamp(std::atoi(mb), 16, 1024) : 64) << 20;
            if (m_pool)
                m_thread = std::thread([this] { Run(); });
        }

        ~ImageMemory()
        {
            {
                std::lock_guard lock(m_mutex);
                m_stop = true;
            }
            m_cv.notify_all();
            if (m_thread.joinable())
                m_thread.join();
            for (auto& spares : m_spares)
                for (const Spare& s : spares)
                    vkFreeMemory(m_device, s.memory, nullptr);
            for (auto& b : m_blocks)
                vkFreeMemory(m_device, b->memory, nullptr);
        }

        // Binds `image` (made, not yet bound) to memory. False, with nothing
        // allocated, when the device has none.
        bool Bind(VkImage image, Allocation& out)
        {
            VkMemoryDedicatedRequirements dedicated{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
            VkMemoryRequirements2 req2{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedicated };
            if (m_pool)
            {
                VkImageMemoryRequirementsInfo2 info{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, image };
                vkGetImageMemoryRequirements2(m_device, &info, &req2);
            }
            else
                vkGetImageMemoryRequirements(m_device, image, &req2.memoryRequirements);
            const VkMemoryRequirements& req = req2.memoryRequirements;
            const uint32_t type = m_findType(req.memoryTypeBits);
            if (type == UINT32_MAX)
                return false;
            out = {};
            out.bytes = req.size;
            if (!m_pool || dedicated.prefersDedicatedAllocation || dedicated.requiresDedicatedAllocation || req.size > m_blockSize / 2 ||
                !BindToBlock(out, req, type))
            {
                VkMemoryDedicatedAllocateInfo dai{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, image };
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.pNext = dedicated.requiresDedicatedAllocation ? &dai : nullptr;
                mai.allocationSize = req.size;
                mai.memoryTypeIndex = type;
                auto start = std::chrono::steady_clock::now();
                if (vkAllocateMemory(m_device, &mai, nullptr, &out.memory) != VK_SUCCESS)
                    return false;
                m_stats.allocations++;
                m_stats.allocationNs += SinceNs(start);
                m_ownBytes += req.size;
                m_own++;
            }
            if (vkBindImageMemory(m_device, image, out.memory, out.offset) != VK_SUCCESS)
            {
                Free(out);
                return false;
            }
            return true;
        }

        // Nothing the GPU may still run can use it: its memory, or its range
        // of a block, is free for reuse at once.
        void Free(Allocation& a)
        {
            if (auto* block = static_cast<Block*>(a.block))
            {
                if (!block->ranges.Free(a.offset, a.bytes))
                {
                    fprintf(stderr, "[rhi/vulkan] image memory: %llu bytes at %llu aren't in use in their block\n", (unsigned long long)a.bytes,
                        (unsigned long long)a.offset);
                    abort();
                }
                if (block->ranges.Empty())
                {
                    // A spare now: the thread frees it if it stays unused beside another.
                    Spare spare{ block->memory, std::chrono::steady_clock::now() };
                    const uint32_t type = block->type;
                    std::erase_if(m_blocks, [block](const std::unique_ptr<Block>& b) { return b.get() == block; });
                    {
                        std::lock_guard lock(m_mutex);
                        m_spares[type].push_back(spare);
                    }
                    m_cv.notify_one();
                }
            }
            else if (a.memory)
            {
                vkFreeMemory(m_device, a.memory, nullptr);
                m_ownBytes -= a.bytes;
                m_own--;
            }
            a = {};
        }

        void Made(uint64_t ns)
        {
            m_stats.images++;
            m_stats.imageNs += ns;
        }

        ImageMemoryStats Stats() const
        {
            ImageMemoryStats s = m_stats;
            for (const auto& b : m_blocks)
            {
                s.blocks++;
                s.blockBytes += b->ranges.Size();
                s.rangeBytes += b->ranges.Used();
                s.ranges += b->ranges.Ranges();
            }
            s.own = m_own;
            s.ownBytes = m_ownBytes;
            {
                std::lock_guard lock(m_mutex);
                for (const auto& spares : m_spares)
                {
                    s.spares += uint32_t(spares.size());
                    s.spareBytes += spares.size() * m_blockSize;
                }
                s.spareBlocks = m_spareBlocks;
                s.spareNs = m_spareNs;
                s.blocksFreed = m_blocksFreed;
            }
            return s;
        }

    private:
        struct Block
        {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t type = 0;
            gpu::BlockRanges ranges;
        };
        struct Spare
        {
            VkDeviceMemory memory;
            std::chrono::steady_clock::time_point since;
        };
        static constexpr auto kSpareKeep = std::chrono::seconds(5);
        static constexpr uint32_t kTypes = VK_MAX_MEMORY_TYPES;

        static uint64_t SinceNs(std::chrono::steady_clock::time_point start)
        {
            return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        }

        VkDeviceMemory AllocateBlock(uint32_t type)
        {
            VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            mai.allocationSize = m_blockSize;
            mai.memoryTypeIndex = type;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            return vkAllocateMemory(m_device, &mai, nullptr, &memory) == VK_SUCCESS ? memory : VK_NULL_HANDLE;
        }

        // nfsmw-memory: makes a spare when one is wanted, and frees spares
        // beyond one once they have been empty kSpareKeep.
        void Run()
        {
            if (m_onThread)
                m_onThread("nfsmw-memory");
            std::unique_lock lock(m_mutex);
            while (!m_stop)
            {
                auto now = std::chrono::steady_clock::now();
                auto next = now + std::chrono::hours(1);
                bool worked = false;
                for (uint32_t type = 0; type < kTypes; type++)
                {
                    std::vector<Spare>& spares = m_spares[type];
                    if (spares.size() > 1 && now - spares.front().since >= kSpareKeep)
                    {
                        VkDeviceMemory memory = spares.front().memory;
                        spares.erase(spares.begin());
                        m_blocksFreed++;
                        lock.unlock();
                        vkFreeMemory(m_device, memory, nullptr);
                        lock.lock();
                        worked = true;
                        break;
                    }
                    if (spares.size() > 1)
                        next = std::min(next, spares.front().since + kSpareKeep);
                    const uint32_t bit = 1u << type;
                    if ((m_spareWanted & bit) && !(m_spareFailed & bit) && spares.empty())
                    {
                        lock.unlock();
                        auto start = std::chrono::steady_clock::now();
                        VkDeviceMemory memory = AllocateBlock(type);
                        const uint64_t ns = SinceNs(start);
                        lock.lock();
                        m_spareWanted &= ~bit;
                        if (!memory)
                        {
                            // Left to the binding thread, which tries again
                            // (and falls back to an allocation per image).
                            m_spareFailed |= bit;
                            fprintf(stderr, "[rhi/vulkan] image memory: no spare block of memory type %u (%llu MB)\n", type,
                                (unsigned long long)(m_blockSize >> 20));
                        }
                        else
                        {
                            m_spareBlocks++;
                            m_spareNs += ns;
                            m_spares[type].push_back({ memory, std::chrono::steady_clock::now() });
                        }
                        worked = true;
                        break;
                    }
                    if (m_spareWanted & bit)
                        m_spareWanted &= ~bit;  // (a spare came back meanwhile, or the thread can't make one)
                }
                if (!worked)
                    m_cv.wait_until(lock, next);
            }
        }

        // A block of `type` for a new image: a spare (the newest; the thread
        // makes the next when it was the last), or one allocated here.
        Block* NewBlock(uint32_t type)
        {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            {
                std::lock_guard lock(m_mutex);
                std::vector<Spare>& spares = m_spares[type];
                if (!spares.empty())
                {
                    memory = spares.back().memory;
                    spares.pop_back();
                }
                if (spares.empty())
                    m_spareWanted |= 1u << type;
            }
            m_cv.notify_one();
            if (memory)
                m_stats.fromSpare++;
            else
            {
                auto start = std::chrono::steady_clock::now();
                memory = AllocateBlock(type);
                m_stats.allocations++;
                m_stats.allocationNs += SinceNs(start);
                if (!memory)
                    return nullptr;
                m_stats.blocksHere++;
                std::lock_guard lock(m_mutex);
                m_spareFailed &= ~(1u << type);
            }
            m_blocks.push_back(std::make_unique<Block>(Block{ memory, type, gpu::BlockRanges(m_blockSize) }));
            return m_blocks.back().get();
        }

        // Binds into a block of `type`; false when no block can be had.
        bool BindToBlock(Allocation& a, const VkMemoryRequirements& req, uint32_t type)
        {
            Block* block = nullptr;
            VkDeviceSize offset = gpu::BlockRanges::kNone;
            for (auto& b : m_blocks)
                if (b->type == type && (offset = b->ranges.Allocate(req.size, req.alignment)) != gpu::BlockRanges::kNone)
                {
                    block = b.get();
                    break;
                }
            if (!block)
            {
                block = NewBlock(type);
                if (!block || (offset = block->ranges.Allocate(req.size, req.alignment)) == gpu::BlockRanges::kNone)
                    return false;
            }
            a.block = block;
            a.offset = offset;
            a.memory = block->memory;
            return true;
        }

        VkDevice m_device;
        std::function<uint32_t(uint32_t)> m_findType;
        std::function<void(const char*)> m_onThread;
        bool m_pool = true;
        VkDeviceSize m_blockSize = 64ull << 20;
        std::vector<std::unique_ptr<Block>> m_blocks;   // in use (the binding thread)
        ImageMemoryStats m_stats;                        // the binding thread's
        uint64_t m_ownBytes = 0;
        uint32_t m_own = 0;
        // The spare thread and the binding thread share these.
        mutable std::mutex m_mutex;
        std::condition_variable m_cv;
        std::vector<Spare> m_spares[kTypes];
        uint32_t m_spareWanted = 0, m_spareFailed = 0;  // memory type bits
        uint64_t m_spareBlocks = 0, m_spareNs = 0, m_blocksFreed = 0;
        bool m_stop = false;
        std::thread m_thread;
    };
}
