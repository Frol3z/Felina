#define PI 3.14159265358979323846

struct VertexOutput
{
    float4 position : SV_Position;
    float3 localPosition : TEXCOORD0;
};

[[vk::binding(0, 0)]]
SamplerState hdrSampler;
[[vk::binding(0, 0)]]
Texture2D hdrEquirectMap;

float2 EquirectangularUV(float3 dir)
{
    // Extract spherical coordinates params for dir
    float theta = atan2(dir.z, dir.x);
    float phi = asin(dir.y);

    // Convert spherical params to the appropriate range
    float u = theta / (2.0 * PI) + 0.5;
    float v = 0.5 - phi / PI;

    return float2(u, v);
}

float4 main(VertexOutput vertIn) : SV_TARGET0
{
    float3 dir = normalize(vertIn.localPosition);
    float2 uv = EquirectangularUV(dir);
    return hdrEquirectMap.Sample(hdrSampler, uv);
}