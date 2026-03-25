struct VertexOutput
{
    float4 position : SV_Position;
    float3 localPosition : TEXCOORD0;
};

struct PushConsts
{
    float4x4 proj;
    float4x4 view;
};

static const float3 cubeVertices[36] =
{
    // +X
    float3(1, -1, -1), float3(1, -1, 1), float3(1, 1, 1),
    float3(1, 1, 1), float3(1, 1, -1), float3(1, -1, -1),

    // -X
    float3(-1, -1, 1), float3(-1, -1, -1), float3(-1, 1, -1),
    float3(-1, 1, -1), float3(-1, 1, 1), float3(-1, -1, 1),

    // +Y
    float3(-1, 1, -1), float3(1, 1, -1), float3(1, 1, 1),
    float3(1, 1, 1), float3(-1, 1, 1), float3(-1, 1, -1),

    // -Y
    float3(-1, -1, 1), float3(1, -1, 1), float3(1, -1, -1),
    float3(1, -1, -1), float3(-1, -1, -1), float3(-1, -1, 1),

    // -Z
    float3(1, -1, 1), float3(1, 1, 1), float3(-1, 1, 1),
    float3(-1, 1, 1), float3(-1, -1, 1), float3(1, -1, 1),

    // +Z
    float3(-1, -1, -1), float3(-1, 1, -1), float3(1, 1, -1),
    float3(1, 1, -1), float3(1, -1, -1), float3(-1, -1, -1)
};

[[vk::push_constant]] PushConsts pushConsts;

VertexOutput main(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float3 pos = cubeVertices[vertexId];
    
    output.position = mul(mul(pushConsts.proj, pushConsts.view), float4(pos, 1.0));
    output.localPosition = pos;
    
    return output;
}