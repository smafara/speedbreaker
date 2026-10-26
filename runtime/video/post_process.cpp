// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include <stdafx.h>
#include "post_process.h"

#include <gpu/renderer.h>
#include <user/settings.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "display_gamma.h"
#include "display_tone.h"
#include "fsr1.h"

namespace video
{
    namespace
    {
        using namespace gpu::rhi;

        const char* kFullscreenVs = R"(#version 460
layout(location = 0) out vec2 uv;
void main()
{
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

        const char* kAaCs = R"(#version 460
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D src;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D dst;
// band: the 16:9 band's columns x..y and rows z..w (video::BandOf), used
// when bandOn is 1 (video/band_filter.h).
layout(push_constant) uniform P { ivec2 size; float subpixel; uint bandOn; ivec4 band; } p;

const float kEdgeMin = 0.0312;      // no AA on darker contrast than this
const float kEdgeRelative = 0.125;  // ... or than this fraction of the local maximum

// This pixel's span of the band's three along each axis, as uv: every fetch
// stays between its first and last texel centres (band_filter.h's Clamp).
vec2 spanLo = vec2(0.0), spanHi = vec2(1.0);

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
vec3 at(vec2 uv)
{
    if (p.bandOn != 0u)
        uv = clamp(uv, spanLo, spanHi);
    return textureLod(src, uv, 0.0).rgb;
}

// band_filter.h's SpanOf: [0, lo), [lo, hi) or [hi, size) holding t.
vec2 spanOf(float t, int lo, int hi, int size)
{
    return t < float(lo) ? vec2(0.0, float(lo)) : t < float(hi) ? vec2(float(lo), float(hi)) : vec2(float(hi), float(size));
}

void main()
{
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= p.size.x || pix.y >= p.size.y)
        return;
    vec2 texel = 1.0 / vec2(p.size);
    vec2 uv = (vec2(pix) + 0.5) * texel;
    if (p.bandOn != 0u)
    {
        vec2 sx = spanOf(float(pix.x) + 0.5, p.band.x, p.band.y, p.size.x), sy = spanOf(float(pix.y) + 0.5, p.band.z, p.band.w, p.size.y);
        spanLo = (vec2(sx.x, sy.x) + 0.5) * texel;
        spanHi = (vec2(sx.y, sy.y) - 0.5) * texel;
    }
    vec3 center = at(uv);
    float lM = luma(center);
    float lN = luma(at(uv + vec2(0.0, -texel.y))), lS = luma(at(uv + vec2(0.0, texel.y)));
    float lW = luma(at(uv + vec2(-texel.x, 0.0))), lE = luma(at(uv + vec2(texel.x, 0.0)));
    float lMin = min(lM, min(min(lN, lS), min(lW, lE)));
    float lMax = max(lM, max(max(lN, lS), max(lW, lE)));
    float range = lMax - lMin;
    if (range < max(kEdgeMin, lMax * kEdgeRelative))
    {
        imageStore(dst, pix, vec4(center, 1.0));
        return;
    }
    float lNW = luma(at(uv - texel)), lSE = luma(at(uv + texel));
    float lNE = luma(at(uv + vec2(texel.x, -texel.y))), lSW = luma(at(uv + vec2(-texel.x, texel.y)));

    // The edge's orientation: where luma changes most (across rows for a
    // horizontal edge).
    float acrossRows = abs(lNW + lSW - 2.0 * lW) + 2.0 * abs(lN + lS - 2.0 * lM) + abs(lNE + lSE - 2.0 * lE);
    float acrossColumns = abs(lNW + lNE - 2.0 * lN) + 2.0 * abs(lW + lE - 2.0 * lM) + abs(lSW + lSE - 2.0 * lS);
    bool horizontal = acrossRows >= acrossColumns;

    // Which side of this pixel the edge is on: the steeper neighbour.
    float lBefore = horizontal ? lN : lW, lAfter = horizontal ? lS : lE;
    float gBefore = abs(lBefore - lM), gAfter = abs(lAfter - lM);
    bool before = gBefore >= gAfter;
    float threshold = 0.25 * max(gBefore, gAfter);
    float step = horizontal ? texel.y : texel.x;
    float edgeLuma = 0.5 * ((before ? lBefore : lAfter) + lM);
    if (before)
        step = -step;

    // Walk along the edge both ways until the luma leaves the edge's.
    vec2 onEdge = uv;
    if (horizontal) onEdge.y += 0.5 * step; else onEdge.x += 0.5 * step;
    vec2 along = horizontal ? vec2(texel.x, 0.0) : vec2(0.0, texel.y);
    vec2 uvA = onEdge - along, uvB = onEdge + along;
    float dA = luma(at(uvA)) - edgeLuma, dB = luma(at(uvB)) - edgeLuma;
    bool endA = abs(dA) >= threshold, endB = abs(dB) >= threshold;
    const float strides[10] = float[](1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);
    for (int i = 0; i < 10 && !(endA && endB); i++)
    {
        if (!endA) { uvA -= along * strides[i]; dA = luma(at(uvA)) - edgeLuma; endA = abs(dA) >= threshold; }
        if (!endB) { uvB += along * strides[i]; dB = luma(at(uvB)) - edgeLuma; endB = abs(dB) >= threshold; }
    }
    float distA = horizontal ? uv.x - uvA.x : uv.y - uvA.y;
    float distB = horizontal ? uvB.x - uv.x : uvB.y - uv.y;
    bool nearerA = distA < distB;
    float offset = 0.5 - min(distA, distB) / (distA + distB);
    // Only blend from the end whose luma variation goes the way this
    // pixel's does (otherwise this pixel is on the edge's far side).
    if (((nearerA ? dA : dB) < 0.0) == (lM < edgeLuma))
        offset = 0.0;

    // Sub-pixel aliasing (thin lines, single-pixel features).
    float average = (2.0 * (lN + lS + lW + lE) + lNW + lNE + lSW + lSE) / 12.0;
    float sub = clamp(abs(average - lM) / range, 0.0, 1.0);
    sub = (3.0 - 2.0 * sub) * sub * sub;
    offset = max(offset, sub * sub * p.subpixel);

    vec2 sampleUv = uv;
    if (horizontal) sampleUv.y += offset * step; else sampleUv.x += offset * step;
    imageStore(dst, pix, vec4(at(sampleUv), 1.0));
}
)";

        const char* kScaleFs = R"(#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D src;
// band: the 16:9 band's columns x..y and rows z..w in texels (video::BandOf),
// used when bandOn is 1 (video/band_filter.h).
layout(push_constant) uniform P { vec2 size; float sharpen; uint bicubic; vec4 band; uint bandOn; } p;

// The span of the band's three along each axis that this pixel's centre
// falls in, as uv: every tap stays between its first and last texel centres
// (band_filter.h's Clamp), so a strip's pixel reads only the strip.
vec2 spanLo = vec2(0.0), spanHi = vec2(1.0);

vec3 at(vec2 t)
{
    if (p.bandOn != 0u)
        t = clamp(t, spanLo, spanHi);
    return textureLod(src, t, 0.0).rgb;
}

// band_filter.h's SpanOf: [0, lo), [lo, hi) or [hi, size) holding t.
vec2 spanOf(float t, float lo, float hi, float size)
{
    return t < lo ? vec2(0.0, lo) : t < hi ? vec2(lo, hi) : vec2(hi, size);
}

void setSpan()
{
    if (p.bandOn == 0u)
        return;
    vec2 pos = uv * p.size;
    vec2 sx = spanOf(pos.x, p.band.x, p.band.y, p.size.x), sy = spanOf(pos.y, p.band.z, p.band.w, p.size.y);
    spanLo = (vec2(sx.x, sy.x) + 0.5) / p.size;
    spanHi = (vec2(sx.y, sy.y) - 0.5) / p.size;
}

// Catmull-Rom bicubic from 9 bilinear taps: the middle two weights of each
// axis are folded into one bilinear fetch.
vec3 catmullRom(vec2 t, vec2 texel)
{
    vec2 pos = t * p.size;
    vec2 c = floor(pos - 0.5) + 0.5;
    vec2 f = pos - c;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = (c - 1.0) * texel, t12 = (c + w2 / w12) * texel, t3 = (c + 2.0) * texel;
    return at(vec2(t0.x, t0.y)) * w0.x * w0.y + at(vec2(t12.x, t0.y)) * w12.x * w0.y + at(vec2(t3.x, t0.y)) * w3.x * w0.y +
        at(vec2(t0.x, t12.y)) * w0.x * w12.y + at(vec2(t12.x, t12.y)) * w12.x * w12.y + at(vec2(t3.x, t12.y)) * w3.x * w12.y +
        at(vec2(t0.x, t3.y)) * w0.x * w3.y + at(vec2(t12.x, t3.y)) * w12.x * w3.y + at(vec2(t3.x, t3.y)) * w3.x * w3.y;
}

void main()
{
    setSpan();
    if (p.bicubic == 0u)
    {
        color = vec4(at(uv), 1.0);
        return;
    }
    vec2 texel = 1.0 / p.size;
    vec3 c = catmullRom(uv, texel);
    // Sharpen against the bilinear sample, then clamp to the 2x2 texels
    // around the sample: detail gets crisper without halos or ringing.
    vec2 corner = (floor(uv * p.size - 0.5) + 0.5) * texel;
    vec3 a = at(corner), b = at(corner + vec2(texel.x, 0.0)), d = at(corner + vec2(0.0, texel.y)), e = at(corner + texel);
    vec3 lo = min(min(a, b), min(d, e)), hi = max(max(a, b), max(d, e));
    c = clamp(c + p.sharpen * (c - at(uv)), lo, hi);
    color = vec4(c, 1.0);
}
)";

        // NFSMW_FAST_SCALE=1 (default on Apple, below; for phones, where this pass at
        // the screen's size is most of the post-processing): the same
        // scaling from 7 fetches a pixel instead of 14 (above). The 2x2
        // texels around the sample come from one gather per channel, and the
        // bilinear sample sharpened against and Catmull-Rom's middle tap are
        // blends of those 4; the 4 corner taps (at most 1.6% of the weight,
        // 1.5 texels away diagonally) are left out and the rest renormalised.
        // The clamp is the same. (A real frame: at most 3 levels apart.)
        const char* kScaleFastFs = R"(#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D src;
// band, bandOn: as kScaleFs's.
layout(push_constant) uniform P { vec2 size; float sharpen; uint bicubic; vec4 band; uint bandOn; } p;

vec2 spanLo = vec2(0.0), spanHi = vec2(1.0);  // as kScaleFs's

vec3 at(vec2 t)
{
    if (p.bandOn != 0u)
        t = clamp(t, spanLo, spanHi);
    return textureLod(src, t, 0.0).rgb;
}

vec2 spanOf(float t, float lo, float hi, float size)
{
    return t < lo ? vec2(0.0, lo) : t < hi ? vec2(lo, hi) : vec2(hi, size);
}

void main()
{
    vec2 sx = vec2(0.0), sy = vec2(0.0);
    if (p.bandOn != 0u)
    {
        vec2 pos = uv * p.size;
        sx = spanOf(pos.x, p.band.x, p.band.y, p.size.x);
        sy = spanOf(pos.y, p.band.z, p.band.w, p.size.y);
        spanLo = (vec2(sx.x, sy.x) + 0.5) / p.size;
        spanHi = (vec2(sx.y, sy.y) - 0.5) / p.size;
    }
    if (p.bicubic == 0u)
    {
        color = vec4(at(uv), 1.0);
        return;
    }
    vec2 texel = 1.0 / p.size;
    vec2 pos = uv * p.size;
    vec2 c = floor(pos - 0.5) + 0.5;
    vec2 f = pos - c;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 g = w2 / w12;
    vec2 t0 = (c - 1.0) * texel, t12 = (c + g) * texel, t3 = (c + 2.0) * texel;
    // Gathers return (x0,y1), (x1,y1), (x1,y0), (x0,y0).
    vec2 mid = (c + 0.5) * texel;
    vec4 r = textureGather(src, mid, 0), gr = textureGather(src, mid, 1), b = textureGather(src, mid, 2);
    vec3 a = vec3(r.w, gr.w, b.w), e = vec3(r.z, gr.z, b.z), d = vec3(r.x, gr.x, b.x), h = vec3(r.y, gr.y, b.y);
    if (p.bandOn != 0u)
    {
        // The gather's row or column past this pixel's span reads the
        // span's edge one instead (band_filter.h's ClampTexel): rows a e
        // (first) and d h, columns a d (first) and e h.
        vec2 first = c - 0.5;
        if (first.y < sy.x) { a = d; e = h; } else if (first.y + 1.0 >= sy.y) { d = a; h = e; }
        if (first.x < sx.x) { a = e; d = h; } else if (first.x + 1.0 >= sx.y) { e = a; h = d; }
    }
    vec3 middle = mix(mix(a, e, g.x), mix(d, h, g.x), g.y);
    vec3 bilinear = mix(mix(a, e, f.x), mix(d, h, f.x), f.y);
    float wN = w12.x * w0.y, wW = w0.x * w12.y, wM = w12.x * w12.y, wE = w3.x * w12.y, wS = w12.x * w3.y;
    vec3 cr = (at(vec2(t12.x, t0.y)) * wN + at(vec2(t0.x, t12.y)) * wW + middle * wM + at(vec2(t3.x, t12.y)) * wE +
        at(vec2(t12.x, t3.y)) * wS) / (wN + wW + wM + wE + wS);
    vec3 lo = min(min(a, e), min(d, h)), hi = max(max(a, e), max(d, h));
    color = vec4(clamp(cr + p.sharpen * (cr - bilinear), lo, hi), 1.0);
}
)";
        // NFSMW_FAST_SCALE: on by default on Apple (iPad: scale pass 1.26 ->
        // 1.07 ms, the same picture within 3 levels) and arm64 Linux (Steam
        // Frame: post-processing 0.47 -> 0.39 ms), off elsewhere until
        // measured there; =0/=1 overrides.
#if defined(__APPLE__) || (defined(__linux__) && defined(__aarch64__))
        constexpr bool kFastScaleByDefault = true;
#else
        constexpr bool kFastScaleByDefault = false;
#endif
        const bool s_fastScale = [] { const char* v = std::getenv("NFSMW_FAST_SCALE"); return v ? v[0] == '1' : kFastScaleByDefault; }();

        // Brightness and the 360's display gamma (video/display_tone.h): the
        // scale shader in use, its main renamed scaleMain, then this main,
        // which looks each channel of its result up in the tone table (256
        // entries for the values k/255, linear between them). The tables
        // are made on the CPU. With Brightness 0 and the 360's gamma off the
        // plain scale pipeline draws, so that picture (v0.1.3's) is the same,
        // bit for bit, as without this.
        const char* kToneGlsl = R"(
layout(set = 0, binding = 1) uniform sampler2D toneTable;

vec3 toneOf(vec3 c)
{
    vec3 t = clamp(c, 0.0, 1.0) * (255.0 / 256.0) + (0.5 / 256.0);
    return vec3(textureLod(toneTable, vec2(t.r, 0.5), 0.0).r, textureLod(toneTable, vec2(t.g, 0.5), 0.0).g,
        textureLod(toneTable, vec2(t.b, 0.5), 0.0).b);
}

void main()
{
    scaleMain();
    color = vec4(toneOf(color.rgb), color.a);
}
)";

        // The tone variant of a scale shader's source ("" if it hasn't
        // exactly one main to rename).
        std::string ToneVariant(const char* scaleGlsl)
        {
            std::string glsl = scaleGlsl;
            const std::string_view kMain = "void main()";
            size_t at = glsl.find(kMain);
            if (at == std::string::npos || glsl.find(kMain, at + 1) != std::string::npos)
                return {};
            glsl.replace(at, kMain.size(), "void scaleMain()");
            return glsl + kToneGlsl;
        }


        // The shaders' push constants (their P in video/fsr1.cpp and above).
        struct EasuPush
        {
            uint32_t con[16];  // FsrEasuCon's con0..con3
            int32_t band[4];   // the 16:9 band's columns x0..x1 and rows y0..y1 (video::BandOf)
            int32_t outW, outH, srcW, srcH;
            uint32_t bandOn;
        };
        static_assert(sizeof(EasuPush) == 100);
        struct RcasPush
        {
            uint32_t con[4];  // FsrRcasCon's (con[0] 0: no sharpening)
            int32_t band[4];
            int32_t originX, originY, outW, outH, srcW, srcH;
            uint32_t bandOn;
        };
        static_assert(sizeof(RcasPush) == 60);
        struct AaPush
        {
            int32_t width, height;
            float subpixel;
            uint32_t bandOn;
            int32_t band[4];
        };
        static_assert(sizeof(AaPush) == 32);
        struct ScalePush
        {
            float width, height, sharpen;
            uint32_t bicubic;
            float band[4];
            uint32_t bandOn;
        };
        static_assert(sizeof(ScalePush) == 36);

        // SPIR-V of `glsl` (stage 0 vertex, 1 fragment, 2 compute); empty if it doesn't compile.
        std::vector<uint32_t> Compile(const std::string& glsl, int stage)
        {
            std::string log;
            std::vector<uint32_t> spirv = gpu::renderer::CompileShader(glsl, stage, log);
            if (spirv.empty())
                fprintf(stderr, "[video] post-processing shader failed to compile:\n%s\n", log.c_str());
            return spirv;
        }

        ShaderCode Code(const std::vector<uint32_t>& spirv) { return { spirv.data(), spirv.size() }; }

        BindingLayoutDesc PushedLayout(std::vector<BindingSlot> slots, uint32_t pushBytes)
        {
            BindingLayoutDesc desc;
            desc.tables.push_back({ std::move(slots), true });
            desc.pushConstantBytes = pushBytes;
            return desc;
        }

        BindingSlot Slot(uint32_t index, BindingKind kind, ShaderStage stage)
        {
            BindingSlot slot;
            slot.index = index;
            slot.kind = kind;
            slot.stages = uint8_t(stage);
            return slot;
        }

        BindingEntry Sampled(uint32_t slot, Image& image, Sampler& sampler)
        {
            BindingEntry entry;
            entry.slot = slot;
            entry.image = { &image, &sampler };
            return entry;
        }

        BindingEntry Stored(uint32_t slot, Image& image)
        {
            BindingEntry entry;
            entry.slot = slot;
            entry.image = { &image, nullptr };
            return entry;
        }

        std::unique_ptr<Image> MakeImage(Device& device, Format format, uint32_t width, uint32_t height, uint32_t usage, const char* name)
        {
            ImageDesc desc;
            desc.format = format;
            desc.extent = { width, height, 1 };
            desc.usage = usage;
            desc.name = name;
            return device.CreateImage(desc);
        }

        void Move(CommandList& list, Image& image, State before, State after)
        {
            const ImageTransition transition{ &image, before, after };
            list.Transition(&transition, 1);
        }
    }

    struct PostProcessor::Impl
    {
        Device& device;
        std::function<void()> waitIdle;
        std::unique_ptr<Sampler> sampler;  // linear, clamped: every sampled read

        // The pipelines and their layouts.
        bool tried = false, ready = false;
        Format format = Format::Undefined;  // the target format the graphics pipelines are made for
        std::vector<uint32_t> vsCode, scaleCode, toneCode, aaCode;
        std::unique_ptr<BindingLayout> aaLayout, scaleLayout, toneLayout;
        std::unique_ptr<Pipeline> aa, scale, toneScale;
        Format scaleMadeFor = Format::Undefined, toneMadeFor = Format::Undefined;

        // The anti-aliased frame, at the game's resolution.
        std::unique_ptr<Image> aaImage;
        uint32_t aaWidth = 0, aaHeight = 0;
        State aaState = State::Undefined;

        // The tone table: 256 x 1 RGBA16F, written from a host-visible buffer.
        bool toneTried = false, toneReady = false;
        std::unique_ptr<Image> toneImage;
        std::unique_ptr<Buffer> toneStaging;
        State toneState = State::Undefined;
        bool toneFilled = false;
        int32_t toneBrightness = 0;
        bool toneRamp = false;
        uint64_t toneRampGeneration = 0;

        // FSR 1 (video/fsr1.h; when it is used: video/upscale.h): EASU
        // (compute) from the anti-aliased frame into fsrImage, the picture's size
        // on the screen, then RCAS (fragment) where the bicubic pass draws.
        // Made the first time a frame is enlarged with it; if that fails, those
        // frames take the bicubic pass (logged once).
        bool fsrTried = false, fsrReady = false, rcasToneTried = false;
        std::vector<uint32_t> easuCode, rcasCode, rcasToneCode;
        std::unique_ptr<BindingLayout> easuLayout, rcasLayout;
        std::unique_ptr<Pipeline> easu, rcas, rcasTone;
        Format rcasMadeFor = Format::Undefined, rcasToneMadeFor = Format::Undefined;
        std::unique_ptr<Image> fsrImage;
        uint32_t fsrWidth = 0, fsrHeight = 0;
        State fsrState = State::Undefined;

        Impl(Device& d, std::function<void()> w) : device(d), waitIdle(std::move(w)) {}

        std::unique_ptr<Pipeline> MakeScalePipeline(const std::vector<uint32_t>& fs, const BindingLayout& layout)
        {
            GraphicsPipelineDesc desc;
            desc.vertex = Code(vsCode);
            desc.fragment = Code(fs);
            desc.layout = &layout;
            desc.colorFormats[0] = format;
            return device.CreateGraphicsPipeline(desc);
        }

        std::unique_ptr<Pipeline> MakeComputePipeline(const std::vector<uint32_t>& code, const BindingLayout& layout)
        {
            ComputePipelineDesc desc;
            desc.code = Code(code);
            desc.layout = &layout;
            return device.CreateComputePipeline(desc);
        }

        // The scale pipeline for the target's format (made again if it changes).
        bool EnsureScalePipeline()
        {
            if (scale && scaleMadeFor == format)
                return true;
            scale = MakeScalePipeline(scaleCode, *scaleLayout);
            if (!scale)
                return false;
            scaleMadeFor = format;
            return true;
        }

        bool EnsurePostProcessing()
        {
            if (tried)
                return ready && EnsureScalePipeline();
            tried = true;
            SamplerDesc sd;
            sd.mip = Filter::Nearest;
            sd.u = sd.v = sd.w = AddressMode::ClampToEdge;
            sampler = device.CreateSampler(sd);
            aaLayout = device.CreateBindingLayout(PushedLayout(
                { Slot(0, BindingKind::CombinedImageSampler, ShaderStage::Compute), Slot(1, BindingKind::StorageImage, ShaderStage::Compute) },
                sizeof(AaPush)));
            scaleLayout = device.CreateBindingLayout(PushedLayout(
                { Slot(0, BindingKind::CombinedImageSampler, ShaderStage::Fragment) }, sizeof(ScalePush)));
            aaCode = Compile(kAaCs, 2);
            vsCode = Compile(kFullscreenVs, 0);
            scaleCode = Compile(s_fastScale ? kScaleFastFs : kScaleFs, 1);
            if (aaCode.empty() || vsCode.empty() || scaleCode.empty())
                return false;
            aa = MakeComputePipeline(aaCode, *aaLayout);
            if (!aa || !EnsureScalePipeline())
                return false;
            ready = true;
            const int32_t mode = settings::GetInt(settings::Id::Scaling);
            fprintf(stderr, "[video] post-processing: %s, %s scaling%s%s\n", AntiAliasing() ? "anti-aliasing" : "no anti-aliasing",
                mode == int32_t(settings::ScalingMode::Fsr1) ? "FSR 1 (enlarging; else bicubic)"
                : mode == int32_t(settings::ScalingMode::Bicubic) ? "bicubic" : "linear",
                mode != int32_t(settings::ScalingMode::Linear) ? std::format(", sharpen {:.2f}", Sharpen()).c_str() : "",
                s_fastScale ? " (NFSMW_FAST_SCALE: 7 fetches)" : "");
            return true;
        }

        void EnsureAaImage(uint32_t width, uint32_t height)
        {
            if (aaImage && aaWidth == width && aaHeight == height)
                return;
            waitIdle();
            aaImage = MakeImage(device, Format::RGBA8, width, height, kImageStorage | kImageSampled, "post anti-aliased");
            aaWidth = width;
            aaHeight = height;
            aaState = State::Undefined;
        }

        // The tone pipeline's pieces, made the first time Brightness or the
        // 360's display gamma is in use (the gamma is on by default, so
        // usually on the first frame).
        bool EnsureTone()
        {
            if (toneTried)
            {
                if (!toneReady)
                    return false;
            }
            else
            {
                toneTried = true;
                std::string glsl = ToneVariant(s_fastScale ? kScaleFastFs : kScaleFs);
                if (glsl.empty())
                {
                    fprintf(stderr, "[video] tone: the scale shader has no single main to wrap; Brightness is off\n");
                    return false;
                }
                toneLayout = device.CreateBindingLayout(PushedLayout(
                    { Slot(0, BindingKind::CombinedImageSampler, ShaderStage::Fragment),
                      Slot(1, BindingKind::CombinedImageSampler, ShaderStage::Fragment) },
                    sizeof(ScalePush)));  // the scale shaders' P
                toneCode = Compile(glsl, 1);
                if (toneCode.empty())
                    return false;
                // 256 x 1 RGBA16F: sampled with linear filtering on every device.
                toneImage = MakeImage(device, Format::RGBA16F, 256, 1, kImageSampled | kImageCopyDst, "tone table");
                BufferDesc bd;
                bd.size = 256 * 4 * sizeof(uint16_t);
                bd.usage = kBufferCopySrc;
                bd.memory = MemoryKind::Upload;
                bd.name = "tone staging";
                toneStaging = device.CreateBuffer(bd);
                if (!toneImage || !toneStaging)
                {
                    fprintf(stderr, "[video] tone: the table couldn't be made; Brightness is off\n");
                    return false;
                }
                toneReady = true;
            }
            if (toneScale && toneMadeFor == format)
                return true;
            auto start = std::chrono::steady_clock::now();
            toneScale = MakeScalePipeline(toneCode, *toneLayout);
            if (!toneScale)
            {
                fprintf(stderr, "[video] tone: the pipeline couldn't be made; Brightness is off\n");
                toneReady = false;
                return false;
            }
            toneMadeFor = format;
            fprintf(stderr, "[video] tone: pipeline made in %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return true;
        }

        bool EnsureFsr()
        {
            if (fsrTried)
                return fsrReady;
            fsrTried = true;
            auto start = std::chrono::steady_clock::now();
            easuLayout = device.CreateBindingLayout(PushedLayout(
                { Slot(0, BindingKind::CombinedImageSampler, ShaderStage::Compute), Slot(1, BindingKind::StorageImage, ShaderStage::Compute) },
                sizeof(EasuPush)));
            // RCAS: EASU's picture, and the tone table (its tone variant).
            rcasLayout = device.CreateBindingLayout(PushedLayout(
                { Slot(0, BindingKind::CombinedImageSampler, ShaderStage::Fragment),
                  Slot(1, BindingKind::CombinedImageSampler, ShaderStage::Fragment) },
                sizeof(RcasPush)));
            easuCode = Compile(fsr1::EasuGlsl(), 2);
            rcasCode = Compile(fsr1::RcasGlsl(), 1);
            if (easuCode.empty() || rcasCode.empty())
            {
                fprintf(stderr, "[video] FSR 1 is off: its shaders didn't compile; bicubic upscaling instead\n");
                return false;
            }
            easu = MakeComputePipeline(easuCode, *easuLayout);
            if (!easu)
            {
                fprintf(stderr, "[video] FSR 1 is off: its EASU pipeline couldn't be made; bicubic upscaling instead\n");
                return false;
            }
            fsrReady = true;
            fprintf(stderr, "[video] FSR 1: EASU and RCAS compiled in %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return true;
        }

        // RCAS's pipeline for the target's format, plain or with the tone
        // table; nullptr if it can't be made (the frame then takes the bicubic pass).
        Pipeline* RcasPipeline(bool tone)
        {
            if (tone && !rcasToneTried)
            {
                rcasToneTried = true;
                const std::string glsl = ToneVariant(fsr1::RcasGlsl().c_str());
                if (!glsl.empty())
                    rcasToneCode = Compile(glsl, 1);
                if (rcasToneCode.empty())
                    fprintf(stderr, "[video] FSR 1: the RCAS shader with Brightness couldn't be made; bicubic upscaling while the tone is in use\n");
            }
            const std::vector<uint32_t>& code = tone ? rcasToneCode : rcasCode;
            std::unique_ptr<Pipeline>& pipeline = tone ? rcasTone : rcas;
            Format& madeFor = tone ? rcasToneMadeFor : rcasMadeFor;
            if (code.empty())
                return nullptr;
            if (pipeline && madeFor == format)
                return pipeline.get();
            pipeline = MakeScalePipeline(code, *rcasLayout);
            madeFor = pipeline ? format : Format::Undefined;
            return pipeline.get();
        }

        void EnsureFsrImage(uint32_t width, uint32_t height)
        {
            if (fsrImage && fsrWidth == width && fsrHeight == height)
                return;
            waitIdle();
            fsrImage = MakeImage(device, Format::RGBA8, width, height, kImageStorage | kImageSampled, "post FSR 1");
            fsrWidth = width;
            fsrHeight = height;
            fsrState = State::Undefined;
        }

        upscale::Pass Choose(int32_t mode, bool tone, uint32_t w, uint32_t h, int32_t dw, int32_t dh)
        {
            upscale::Pass pass = upscale::Choose(mode, w, h, dw, dh);
            if (pass == upscale::Pass::Fsr1 && !(EnsureFsr() && RcasPipeline(tone)))
                pass = upscale::Pass::Bicubic;
            return pass;
        }
    };

    PostProcessor::PostProcessor(Device& device, std::function<void()> waitIdle)
        : m_impl(std::make_unique<Impl>(device, std::move(waitIdle))) {}
    PostProcessor::~PostProcessor() = default;

    bool PostProcessor::Ready(Format format)
    {
        m_impl->format = format;
        return m_impl->EnsurePostProcessing();
    }

    bool PostProcessor::AntiAliasing() { return settings::GetBool(settings::Id::AntiAliasing); }
    float PostProcessor::Sharpen() { return std::clamp(settings::GetFloat(settings::Id::Sharpening), 0.0f, 1.0f); }

    Image& PostProcessor::RecordAntiAliasing(CommandList& list, Image& source, uint32_t w, uint32_t h, const FrameBand& band, bool bandOn)
    {
        Impl& m = *m_impl;
        m.EnsureAaImage(w, h);
        // After the last frame's scale pass read it.
        Move(list, *m.aaImage, m.aaState, State::ShaderWrite);
        list.SetPipeline(*m.aa);
        const BindingEntry entries[2] = { Sampled(0, source, *m.sampler), Stored(1, *m.aaImage) };
        list.PushBindings(*m.aaLayout, 0, entries, 2);
        const AaPush pc = { int32_t(w), int32_t(h), 0.5f, bandOn ? 1u : 0u, { band.x0, band.x1, band.y0, band.y1 } };
        list.PushConstants(*m.aaLayout, &pc, sizeof(pc));
        list.Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        Move(list, *m.aaImage, State::ShaderWrite, State::ShaderRead);
        m.aaState = State::ShaderRead;
        return *m.aaImage;
    }

    bool PostProcessor::PrepareTone(CommandList& list)
    {
        Impl& m = *m_impl;
        const int32_t brightness = std::clamp(settings::GetInt(settings::Id::Brightness), tone::kBrightnessMin, tone::kBrightnessMax);
        const bool useRamp = settings::GetBool(settings::Id::DisplayGamma);
        if (tone::Neutral(brightness, useRamp) || !m.EnsureTone())
            return false;
        const uint64_t rampGeneration = useRamp ? display_gamma::Generation() : 0;
        if (m.toneFilled && m.toneBrightness == brightness && m.toneRamp == useRamp && m.toneRampGeneration == rampGeneration)
            return true;
        display_gamma::Table ramp;
        const bool haveRamp = useRamp && display_gamma::Latest(ramp) != 0;
        const tone::Lut lut = tone::BuildLut(brightness, haveRamp ? &ramp : nullptr);
        const auto texels = tone::ToHalfTexels(lut);
        // The frames in flight may be reading the table: rare (a setting
        // or the game's own ramp changed), so wait for them.
        m.waitIdle();
        memcpy(m.toneStaging->Map(), texels.data(), sizeof(texels));
        Move(list, *m.toneImage, m.toneState, State::CopyDst);
        BufferImageCopy copy;
        copy.region = { 0, 0, 256, 1 };
        list.CopyBufferToImage(*m.toneStaging, *m.toneImage, &copy, 1);
        Move(list, *m.toneImage, State::CopyDst, State::ShaderRead);
        m.toneState = State::ShaderRead;
        fprintf(stderr, "[video] tone: brightness %+d, Xbox 360 display gamma %s; 64->%.1f 128->%.1f 192->%.1f (8-bit levels, red)\n",
            brightness, !useRamp ? "off" : haveRamp ? std::format("on (the game's ramp #{})", rampGeneration).c_str() : "on (no ramp from the game yet)",
            lut.entry[64][0] * 255.0, lut.entry[128][0] * 255.0, lut.entry[192][0] * 255.0);
        m.toneFilled = true;
        m.toneBrightness = brightness;
        m.toneRamp = useRamp;
        m.toneRampGeneration = rampGeneration;
        return true;
    }

    upscale::Pass PostProcessor::PassFor(int32_t mode, bool tone, uint32_t w, uint32_t h, int32_t dw, int32_t dh)
    {
        return m_impl->Choose(mode, tone, w, h, dw, dh);
    }

    upscale::Pass PostProcessor::ChoosePass(bool tone, uint32_t w, uint32_t h, int32_t dw, int32_t dh)
    {
        const upscale::Pass pass = m_impl->Choose(settings::GetInt(settings::Id::Scaling), tone, w, h, dw, dh);
        static upscale::Pass lastPass = upscale::Pass::Linear;
        static uint32_t lastW = 0, lastH = 0;
        static int32_t lastDw = 0, lastDh = 0;
        static int lines = 0;
        if ((pass != lastPass || w != lastW || h != lastH || dw != lastDw || dh != lastDh) && lines < 32)
        {
            lines++;
            const bool fsrChosen = settings::GetInt(settings::Id::Scaling) == int32_t(settings::ScalingMode::Fsr1);
            fprintf(stderr, "[video] upscaling: %s, %ux%u -> %dx%d%s\n", upscale::PassName(pass), w, h, dw, dh,
                pass == upscale::Pass::Fsr1 ? std::format(" (RCAS at {:.2f} of its strength)", upscale::RcasAmount(Sharpen())).c_str()
                : !fsrChosen ? ""
                : upscale::Enlarged(w, h, dw, dh) ? " (FSR 1 unavailable)" : " (FSR 1 only enlarges)");
        }
        lastPass = pass;
        lastW = w;
        lastH = h;
        lastDw = dw;
        lastDh = dh;
        return pass;
    }

    void PostProcessor::RecordEasu(CommandList& list, Image& input, uint32_t w, uint32_t h, int32_t dw, int32_t dh, const FrameBand& band,
        bool bandOn)
    {
        Impl& m = *m_impl;
        m.EnsureFsrImage(uint32_t(dw), uint32_t(dh));
        // After the last frame's RCAS read it.
        Move(list, *m.fsrImage, m.fsrState, State::ShaderWrite);
        list.SetPipeline(*m.easu);
        const BindingEntry entries[2] = { Sampled(0, input, *m.sampler), Stored(1, *m.fsrImage) };
        list.PushBindings(*m.easuLayout, 0, entries, 2);
        EasuPush pc{};
        const std::array<uint32_t, 16> con = fsr1::EasuConstants(w, h, uint32_t(dw), uint32_t(dh));
        std::copy(con.begin(), con.end(), pc.con);
        pc.band[0] = band.x0;
        pc.band[1] = band.x1;
        pc.band[2] = band.y0;
        pc.band[3] = band.y1;
        pc.outW = dw;
        pc.outH = dh;
        pc.srcW = int32_t(w);
        pc.srcH = int32_t(h);
        pc.bandOn = bandOn ? 1u : 0u;
        list.PushConstants(*m.easuLayout, &pc, sizeof(pc));
        list.Dispatch((uint32_t(dw) + 7) / 8, (uint32_t(dh) + 7) / 8, 1);
        Move(list, *m.fsrImage, State::ShaderWrite, State::ShaderRead);
        m.fsrState = State::ShaderRead;
    }

    void PostProcessor::DrawPicture(CommandList& list, upscale::Pass pass, bool tone, Image& input, uint32_t w, uint32_t h,
        const FrameBand& band, bool bandOn, int32_t dx, int32_t dy, int32_t dw, int32_t dh, float sharpen)
    {
        Impl& m = *m_impl;
        list.SetViewport(float(dx), float(dy), float(dw), float(dh), 0.0f, 1.0f);
        list.SetScissor({ dx, dy, uint32_t(dw), uint32_t(dh) });
        const bool fsr = pass == upscale::Pass::Fsr1;
        const BindingLayout& layout = fsr ? *m.rcasLayout : tone ? *m.toneLayout : *m.scaleLayout;
        list.SetPipeline(fsr ? *m.RcasPipeline(tone) : tone ? *m.toneScale : *m.scale);
        BindingEntry entries[2] = { Sampled(0, fsr ? *m.fsrImage : input, *m.sampler) };
        if (tone)
            entries[1] = Sampled(1, *m.toneImage, *m.sampler);
        list.PushBindings(layout, 0, entries, tone ? 2 : 1);
        if (fsr)
        {
            RcasPush pc{};
            const std::array<uint32_t, 4> con = fsr1::RcasConstants(upscale::RcasStops(sharpen));
            std::copy(con.begin(), con.end(), pc.con);
            pc.band[0] = band.x0;
            pc.band[1] = band.x1;
            pc.band[2] = band.y0;
            pc.band[3] = band.y1;
            pc.originX = dx;
            pc.originY = dy;
            pc.outW = dw;
            pc.outH = dh;
            pc.srcW = int32_t(w);
            pc.srcH = int32_t(h);
            pc.bandOn = bandOn ? 1u : 0u;
            list.PushConstants(layout, &pc, sizeof(pc));
        }
        else
        {
            const ScalePush pc = { float(w), float(h), sharpen, pass == upscale::Pass::Bicubic ? 1u : 0u,
                { float(band.x0), float(band.x1), float(band.y0), float(band.y1) }, bandOn ? 1u : 0u };
            list.PushConstants(layout, &pc, sizeof(pc));
        }
        list.Draw(3, 1, 0, 0);
    }
}
