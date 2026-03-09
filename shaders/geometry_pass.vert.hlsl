// Camera uniform buffer
struct CameraData
{
    float3 position;
    float4x4 view;
    float4x4 proj;
    float4x4 invViewProj;
    float skyboxIntensity;
    float exposure;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CameraData> cameraData;

// Per-object data
struct ObjectData
{
    float4x4 model;
    float3x3 normal;
};

[[vk::binding(0, 1)]]
StructuredBuffer<ObjectData> objectBuffer;

// Push constant used to access the objectBuffer
struct PushConsts
{
    uint objectIndex;
    uint materialIndex;
};
[[vk::push_constant]] PushConsts pushConsts;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint materialIndex : TEXCOORD1;
};

VertexOutput main(VertexInput input, uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float4x4 model = objectBuffer[pushConsts.objectIndex].model;
    float3x3 normalMatrix = objectBuffer[pushConsts.objectIndex].normal;
    
    output.position = mul(cameraData.proj, mul(cameraData.view, mul(model, float4(input.position, 1.0)))); // canonical view-volume
    output.normal = normalize(mul(normalMatrix, input.normal)); // world-space normal
    output.uv = input.uv;
    output.materialIndex = pushConsts.materialIndex;
    return output;
}