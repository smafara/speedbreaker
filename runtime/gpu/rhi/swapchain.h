// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The window's pictures: a surface, the swapchain on it, and what the
// presenter's pacing policy needs to know about each present. See device.h.
//
// DRAFT for review (DECISIONS.md D11): nothing includes this yet.
//
// WHAT CROSSES THE BOUNDARY, found by reading video/presenter.cpp:
//
//  * Down: an image to draw to, and per present: an id the caller picked,
//    and optionally "not before this time" (macOS shows a frame at a set
//    time after its guest vblank: VK_GOOGLE_display_timing's
//    desiredPresentTime on MoltenVK, presentAtTime: on Metal).
//  * Up, per present: when it reached the screen (the guest vblank locks to
//    the display's refresh from these: RefreshFit, VblankLock), and the
//    driver's own report on a frame shown at a time (the watchdog in
//    present_schedule.h).
//
//  Vulkan learns the first by waiting for the present on a thread
//  (VK_KHR_present_wait) and the second by polling
//  vkGetPastPresentationTimingGOOGLE; Metal has a presented handler on the
//  drawable that gives both at once. So the interface is NOT the Vulkan
//  calls: it is a completion CALLBACK, which each backend feeds its own way.
//  The waiter thread, its QoS, the generation scheme that keeps it off a
//  destroyed swapchain (video/swapchain_use.h) and the clock conversions
//  (Metal's host time, mach_absolute_time) all go inside the Vulkan backend.
//
//  What stays in the presenter (policy, shared): the hold before a present
//  (VblankLock::PresentAt: sleeps, then presents), the choice of the time
//  (present_schedule.h, hfr_schedule.h), the refresh fit, the repeat count,
//  the swapchain policy (swapchain_policy.h: when to rebuild and at what
//  size), the iOS present scale and SDL.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "types.h"

namespace gpu::rhi
{
    class Image;

    // A window's drawing surface. Made by the backend from the platform's
    // window (vulkan_native.h: from a VkSurfaceKHR; metal_native.h: from a
    // CAMetalLayer), which the presenter gets from SDL.
    class Surface
    {
    public:
        virtual ~Surface() = default;
        // Its size now, in pixels; {0, 0}: minimised. Vulkan: currentExtent,
        // or (Wayland) the window's, which the presenter passes at creation.
        virtual Extent3D CurrentExtent() = 0;
    };

    enum class SurfaceStatus : uint8_t
    {
        Ok,
        Suboptimal,   // works; build it again when convenient
        OutOfDate,    // the image wasn't drawn to / presented: build it again
    };

    // 0: unknown. Steady-clock nanoseconds throughout (std::chrono::steady_clock,
    // what the presenter's NowNs() reads); a backend converts from its own clock.
    using TimeNs = int64_t;

    // A present reached the screen. Delivered in present order, one at a time,
    // from a thread of the backend's (never the caller's).
    struct PresentShown
    {
        uint64_t id = 0;
        TimeNs presented = 0;    // when Present was called
        TimeNs onScreen = 0;     // when it was seen to be on screen
        // True when `onScreen` was measured as it happened. False: caught up
        // on afterwards (a wait that began after the present was long shown),
        // so it says nothing of the display's timing. A backend with the
        // display's own time (Metal) always says true.
        bool timed = false;
    };

    // The driver's report on a present asked to be shown at a time
    // (PresentRequest::notBefore). Arrives some presents later, or never.
    struct PresentReport
    {
        uint64_t id = 0;
        TimeNs onScreen = 0;     // 0: it was never shown (a dropped drawable)
    };

    struct SwapchainDesc
    {
        Extent3D extent;                 // from the policy (it may differ from the surface's: iOS portrait)
        Format preferredFormat = Format::BGRA8;   // sRGB-encoded, non-linear; the backend picks the nearest it has
        bool vsync = true;               // false: the newest picture wins where the system has that (mailbox), else tears
        uint32_t minImages = 3;          // MoltenVK: 3, so a present doesn't wait for the display to free a drawable
        bool readable = false;           // CopySrc wanted (screenshots, NFSMW_CHECK_PRESENT_SEC); see Swapchain::Readable
        std::function<void(const PresentShown&)> onShown;      // empty: presents aren't timed
        std::function<void(const PresentReport&)> onReport;    // empty: not wanted
        bool reportAll = false;          // reports for every present, not only the ones with a notBefore (diagnostics)
        // Runs first on the backend's callback thread: the caller names it
        // (hang reports) and sets its priority; the backend knows neither.
        std::function<void()> onThreadStart;
    };

    // One request to show the image acquired last.
    struct PresentRequest
    {
        // The caller's, increasing; echoed in the callbacks. 0: no callbacks
        // for this one (a Swapchain without Capabilities::presentTiming
        // ignores ids).
        uint64_t id = 0;
        // Show it no earlier than this (steady clock); 0: when ready. Only
        // honoured with Capabilities::scheduledPresent.
        TimeNs notBefore = 0;
        // The time Present is called, for PresentShown::presented; 0: the
        // backend's own stamp. (The presenter stamps before its hold.)
        TimeNs presented = 0;
    };

    class Swapchain
    {
    public:
        // Waits until the callbacks have stopped and no present is in flight,
        // and does not return before the display is done with the images.
        virtual ~Swapchain() = default;

        virtual Extent3D Extent() const = 0;
        virtual Format GetFormat() const = 0;
        virtual uint32_t ImageCount() const = 0;
        // Images can be copied from (the surface allowed CopySrc).
        virtual bool Readable() const = 0;
        // For the log: "FIFO", "MAILBOX", "IMMEDIATE", "displaySync".
        virtual const char* PresentModeName() const = 0;
        // The mode and what else the surface offers, for the system summary.
        virtual std::string Describe() const = 0;

        // The next image to draw, usable as RenderTarget and CopyDst (and
        // CopySrc when Readable) and transitioned Undefined -> ... -> Present
        // by the command list that draws to it. Blocks until one is free.
        // The image is the swapchain's: valid until Present or the next
        // Acquire, and the list that draws it must go to the queue with
        // Device::SubmitToSwapchain. nullptr with OutOfDate.
        virtual SurfaceStatus Acquire(Image*& image) = 0;

        // Shows the acquired image, after the submission that draws it.
        // Takes the queue lock inside (the presenter holds none).
        virtual SurfaceStatus Present(const PresentRequest& request) = 0;
    };
}
