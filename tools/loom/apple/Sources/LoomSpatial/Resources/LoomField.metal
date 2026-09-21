#include <metal_stdlib>
using namespace metal;

struct LoomFieldUniforms {
    float4 timing;
    float4 accent;
};

struct LoomFieldVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex LoomFieldVertexOut loom_field_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
        float2(-1.0, 1.0), float2(1.0, -1.0), float2(1.0, 1.0)
    };
    LoomFieldVertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = positions[vertexID] * 0.5 + 0.5;
    return out;
}

float loom_hash(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float loom_noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(loom_hash(i), loom_hash(i + float2(1.0, 0.0)), f.x),
        mix(loom_hash(i + float2(0.0, 1.0)), loom_hash(i + 1.0), f.x),
        f.y
    );
}

float loom_fbm(float2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int octave = 0; octave < 5; ++octave) {
        value += amplitude * loom_noise(p);
        p = p * 2.03 + float2(3.1, 1.7);
        amplitude *= 0.48;
    }
    return value;
}

fragment float4 loom_field_fragment(
    LoomFieldVertexOut in [[stage_in]],
    constant LoomFieldUniforms &uniforms [[buffer(0)]]
) {
    float2 resolution = max(uniforms.timing.xy, float2(1.0));
    float time = uniforms.timing.z;
    float2 uv = (in.uv * resolution - 0.5 * resolution) / resolution.y;

    float2 drift = float2(time * 0.025, -time * 0.018);
    float2 warp = float2(
        loom_fbm(uv * 1.6 + drift),
        loom_fbm(uv * 1.8 - drift + 7.2)
    );
    float volume = loom_fbm(uv * 2.1 + (warp - 0.5) * 1.3 + drift);
    float caustic = pow(saturate(1.0 - abs(sin((uv.x + warp.y) * 8.0) * cos((uv.y - warp.x) * 7.0))), 9.0);

    float horizon = exp(-7.0 * abs(uv.y + 0.13 + 0.08 * sin(uv.x * 3.0 + time * 0.22)));
    float ribbonWave = sin((uv.x + warp.x * 0.42) * 5.2 + (uv.y - warp.y * 0.24) * 3.4 + time * 0.20);
    float ribbon = pow(saturate(1.0 - abs(ribbonWave)), 12.0);
    float latticeX = smoothstep(0.985, 1.0, cos((uv.x + warp.x * 0.08) * 34.0));
    float latticeY = smoothstep(0.988, 1.0, cos((uv.y + warp.y * 0.08) * 34.0));
    float lattice = max(latticeX, latticeY) * 0.13;

    float perspectiveY = 1.0 / max(abs(uv.y + 0.58), 0.09);
    float perspectiveGrid = smoothstep(0.965, 1.0, cos((uv.x * perspectiveY + time * 0.018) * 15.0));
    perspectiveGrid *= (1.0 - smoothstep(-0.52, -0.18, uv.y)) * 0.07;

    // Layered signal filaments give the field depth without competing with the work surface.
    float filaments = 0.0;
    float filamentGlow = 0.0;
    for (int layer = 0; layer < 4; ++layer) {
        float depth = float(layer) / 3.0;
        float frequency = 2.2 + depth * 1.8;
        float speed = 0.055 + depth * 0.035;
        float strand = sin(uv.x * frequency + time * speed + depth * 4.7);
        strand += 0.35 * sin(uv.x * 7.0 - time * speed * 0.7 + warp.x * 2.8);
        float strandY = (strand * (0.055 + depth * 0.025)) - 0.10 + depth * 0.12;
        float distanceToStrand = abs(uv.y - strandY);
        filaments += exp(-distanceToStrand * (115.0 - depth * 24.0)) * (0.22 + depth * 0.16);
        filamentGlow += exp(-distanceToStrand * (22.0 - depth * 5.0)) * 0.045;
    }

    float glassRay = pow(saturate(1.0 - abs(
        sin(uv.x * 2.8 + warp.y * 1.7 + time * 0.025)
    )), 22.0);
    glassRay *= smoothstep(-0.72, 0.32, uv.y) * (1.0 - smoothstep(0.25, 0.88, uv.y));

    float3 base = float3(0.012, 0.018, 0.032);
    float3 cyan = float3(0.16, 0.74, 0.86);
    float3 magenta = float3(0.74, 0.18, 0.57);
    float3 amber = float3(0.95, 0.48, 0.12);
    float3 spectrum = mix(cyan, magenta, smoothstep(0.34, 0.78, volume));
    spectrum = mix(spectrum, amber, caustic * 0.32);
    spectrum = mix(spectrum, uniforms.accent.rgb, 0.22);

    float vignette = 1.0 - smoothstep(0.18, 1.15, length(uv * float2(0.78, 1.0)));
    float intensity = (
        0.09 + volume * 0.20 + caustic * 0.12 + horizon * 0.18 + ribbon * 0.10
        + lattice + perspectiveGrid + filaments * 0.12 + filamentGlow + glassRay * 0.08
    ) * vignette;
    float3 color = base + spectrum * intensity * uniforms.accent.a;
    return float4(color, 1.0);
}
