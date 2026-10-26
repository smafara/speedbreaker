// The Metal swapchain (gpu/rhi/metal): a window's CAMetalLayer, drawables
// acquired and drawn to through the RHI, presents with the display's own
// times, and presents asked to be shown at a time. Opens a small window for
// about a second; skipped without a display.
#include <gpu/rhi/metal/metal_device.h>
#include <gpu/rhi/metal/metal_native.h>
#include <gpu/rhi/command_list.h>

#include <QuartzCore/QuartzCore.hpp>
#include <SDL3/SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>

using namespace gpu::rhi;

static int g_failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); g_failures++; } } while (0)

static int64_t NowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        printf("no display: skipped (%s)\n", SDL_GetError());
        return 0;
    }
    auto device = metal::CreateDevice();
    if (!device)
    {
        printf("no Metal device: skipped\n");
        return 0;
    }
    SDL_Window* window = SDL_CreateWindow("swapchain test", 640, 360, SDL_WINDOW_METAL | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    SDL_MetalView view = window ? SDL_Metal_CreateView(window) : nullptr;
    void* layer = view ? SDL_Metal_GetLayer(view) : nullptr;
    if (!layer)
    {
        printf("no Metal layer: skipped (%s)\n", SDL_GetError());
        return 0;
    }
    SDL_ShowWindow(window);
    SDL_RaiseWindow(window);
    for (int i = 0; i < 50; i++)
    {
        SDL_PumpEvents();
        SDL_Delay(10);
    }
    auto surface = metal::CreateSurface(*device, layer);
    const Extent3D size = surface->CurrentExtent();
    CHECK(size.width > 0 && size.height > 0, "the surface has a size (%u x %u)", size.width, size.height);

    struct Seen { uint64_t id; int64_t presented, onScreen; };
    std::mutex mutex;
    std::vector<Seen> shown;
    struct Report { uint64_t id; int64_t onScreen; };
    std::vector<Report> reports;
    int threadStarts = 0;
    SwapchainDesc sd;
    sd.extent = size;
    sd.readable = true;
    sd.onShown = [&](const PresentShown& s) { std::lock_guard l(mutex); shown.push_back({ s.id, s.presented, s.onScreen }); };
    sd.onReport = [&](const PresentReport& r) { std::lock_guard l(mutex); reports.push_back({ r.id, r.onScreen }); };
    sd.onThreadStart = [&] { std::lock_guard l(mutex); threadStarts++; };
    auto swapchain = device->CreateSwapchain(*surface, sd);
    CHECK(swapchain != nullptr, "swapchain");
    if (!swapchain)
        return 1;
    printf("  %s\n", swapchain->Describe().c_str());
    {
        // The picture is sRGB-encoded: the layer has to be sRGB, as MoltenVK makes it
        // (SDL sets it; nothing here does).
        CGColorSpaceRef space = static_cast<CA::MetalLayer*>(layer)->colorspace();
        CFStringRef name = space ? CGColorSpaceGetName(space) : nullptr;
        CHECK(name && CFEqual(name, kCGColorSpaceSRGB), "the layer's color space is sRGB");
    }
    CHECK(device->GetCapabilities().presentTiming && device->GetCapabilities().scheduledPresent, "timing offered");

    auto list = device->CreateCommandList();
    auto fence = device->CreateFence();
    constexpr int kFrames = 120;
    int64_t base = 0;
    std::vector<int64_t> wanted(kFrames + 1, 0);
    for (int i = 1; i <= kFrames; i++)
    {
        SDL_PumpEvents();
        Image* image = nullptr;
        CHECK(swapchain->Acquire(image) == SurfaceStatus::Ok && image, "acquire %d", i);
        if (!image)
            break;
        CHECK(image->Desc().extent.width == size.width, "the image has the swapchain's size");
        list->Begin();
        ImageTransition in[] = { { image, State::Undefined, State::RenderTarget } };
        list->Transition(in, 1);
        RenderingDesc rd;
        const float v = float(i % 60) / 60.0f;
        rd.color[0] = { image, LoadOp::Clear, StoreOp::Store, { { v, 0.3f, 1.0f - v, 1 } } };
        rd.area = { 0, 0, size.width, size.height };
        list->BeginRendering(rd);
        list->EndRendering();
        ImageTransition out[] = { { image, State::RenderTarget, State::Present } };
        list->Transition(out, 1);
        list->End();
        device->SubmitToSwapchain(*list, *fence, *swapchain);
        fence->Wait();
        fence->Reset();
        PresentRequest pr;
        pr.id = uint64_t(i);
        if (i > 60)  // the second half: a frame every 20 ms, from 50 ms on
        {
            if (i == 61)
                base = NowNs() + 50'000'000;
            pr.notBefore = wanted[i] = base + int64_t(i - 61) * 20'000'000;
        }
        CHECK(swapchain->Present(pr) == SurfaceStatus::Ok, "present %d", i);
    }
    swapchain.reset();  // waits for the reports

    std::lock_guard l(mutex);
    CHECK(int(shown.size()) >= kFrames - 5, "presents were shown (%zu of %d)", shown.size(), kFrames);
    {
        std::vector<bool> seen(kFrames + 1);
        for (const Seen& x : shown)
            seen[x.id] = true;
        for (int i = 6; i <= kFrames; i++)  // (the window's first frames may go unseen while it appears)
            CHECK(seen[i], "frame %d was shown", i);
    }
    CHECK(threadStarts >= 1, "onThreadStart ran");
    bool ordered = true;
    for (size_t i = 1; i < shown.size(); i++)
        ordered &= shown[i].id > shown[i - 1].id && shown[i].onScreen >= shown[i - 1].onScreen;
    CHECK(ordered, "shown in present order");
    std::vector<double> gaps;
    for (size_t i = 10; i < 55 && i < shown.size(); i++)
        gaps.push_back(double(shown[i].onScreen - shown[i - 1].onScreen) * 1e-6);
    if (!gaps.empty())
    {
        std::sort(gaps.begin(), gaps.end());
        printf("  frame gap, first 60 (display-synced): median %.2f ms, min %.2f, max %.2f\n", gaps[gaps.size() / 2], gaps.front(), gaps.back());
        CHECK(gaps[gaps.size() / 2] > 3.0 && gaps[gaps.size() / 2] < 40.0, "a gap of about a refresh");
    }
    int asked = 0, early = 0, late = 0, dropped = 0;
    for (const Report& r : reports)
    {
        if (r.id >= wanted.size() || !wanted[r.id])
            continue;
        asked++;
        early += r.onScreen && r.onScreen < wanted[r.id] - 2'000'000;
        late += r.onScreen > wanted[r.id] + 20'000'000;
        dropped += r.onScreen == 0;
    }
    printf("  %d presents asked for a time: %d reports; early %d, more than 20 ms late %d, dropped %d\n", kFrames - 60, asked, early, late, dropped);
    CHECK(asked == kFrames - 60, "a report for every present with a time");
    CHECK(early == 0 && dropped == 0, "none shown before its time, none dropped");
    CHECK(late <= 2, "shown within a refresh or two of its time");
    if (asked > 2)
    {
        const int64_t span = shown.back().onScreen - shown[shown.size() - (kFrames - 60)].onScreen;
        printf("  the scheduled frames spanned %.1f ms for %d gaps of 20 ms\n", double(span) * 1e-6, kFrames - 61);
        CHECK(std::abs(double(span) * 1e-6 - 20.0 * (kFrames - 61)) < 30.0, "their spacing follows the asked one");
    }
    SDL_Metal_DestroyView(view);
    SDL_DestroyWindow(window);
    SDL_Quit();
    printf(g_failures ? "%d FAILED\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
