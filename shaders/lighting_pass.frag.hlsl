#define MAX_TEXTURES 136 // must match the one in Renderer.hpp
#define MAX_SAMPLERS 2
#define MAX_LIGHTS 16
#define PI 3.14159265358979323846

// Data type definitions
// TODO: maybe move into a dedicated file
struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct CameraData
{
    float3      position;
    float4x4    view;
    float4x4    proj;
    float4x4    invViewProj;
};

// NOTE: .w for color, position and direction is padding
struct LightData // 80 byte
{
    uint    type;           // DIRECTIONAL = 0, POINT = 1, SPOT = 2    
    float4  color;
    float4  position;       // used by POINT and SPOT
    float4  direction;      // used by DIRECTIONAL and SPOT
    float   intensity;
    float   range;
    float   innerConeAngle;
    float   outerConeAngle;
};

struct LightsUBO
{
    LightData lights[MAX_LIGHTS]; // ~1kB
    uint lightsCount;
    uint padding[3];
};

// Camera uniform buffer (set 0)
[[vk::binding(0, 0)]]
ConstantBuffer<CameraData> cameraData;

// Lights uniform buffer (set 1)
[[vk::binding(0, 1)]]
ConstantBuffer<LightsUBO> lightsUBO;

// G-buffer (set 2)
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D gBaseColor;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState gBaseColorSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D gMaterialInfo;
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState gMaterialInfoSampler;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D gNormal;
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState gNormalSampler;

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D gDepth;
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState gDepthSampler;

[[vk::binding(0, 3)]]
SamplerState samplers[MAX_SAMPLERS]; // NOTE: currently always defaulting to samplers[0]

[[vk::binding(1, 3)]]
Texture2D textures[MAX_TEXTURES];

[[vk::binding(2, 3)]]
TextureCube skybox;

float3 reconstructWorldPosition(float depth, float2 uv)
{
    float2 ndc = uv * 2.0 - 1.0; // convert [0, 1] to [-1, 1]
    float4 clipPos = float4(ndc.x, ndc.y, depth, 1.0);
    float4 worldPosH = mul(cameraData.invViewProj, clipPos); // clip to world-space
    return (worldPosH.xyz / worldPosH.w); // de-homogenization
}

// Metalness-weighted Lambertian diffuse BRDF
float3 diffuseBRDF(const float3 albedo, const float metalness)
{
    return (1.0 - metalness) * (float3) (1.0 / PI) * albedo;
}

/** 
 *  Schlick's approximation of Fresnel equations
 * 
 *  NOTES:
 *  - hDotV is the cosine of the angle between the 
 *    sampled microfacet normal and the view direction
 *    (clamped between 0 and 1)
 *  - metals should provide the base color as f0, while
 *    dielectrics should use a value of 0.04 as a good
 *    approximation of their behaviour (see Real
 *    Time Rendering)
 *
 * OPTIMIZATIONS:
 * - S. Lagarde, "Spherical Gaussian approximation for
 *   Blinn-Phong, Phong and Fresnel", 2012
**/ 
float3 F(float3 f0, float hDotV)
{
    float3 f90 = float3(1.0, 1.0, 1.0);
    return f0 + (f90 - f0) * pow(1 - hDotV, 5.0);
}

// GGX normal distribution function
float D(float alphaSquared, float nDotH)
{
    float b = ((alphaSquared - 1.0f) * nDotH * nDotH + 1.0f);
    return alphaSquared / (PI * b * b);
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Height correlated version - optimized by substituing 
// G_Lambda for G_Lambda_GGX and dividing by (4 * NdotL * NdotV) to cancel out 
// the terms in specular BRDF denominator
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
// 
// REFERENCE: https://boksajak.github.io/files/CrashCourseBRDF.pdf
// Improved numerical stability
float G(float alpha2, float NdotL, float NdotV)
{
    NdotL = saturate(NdotL);
    NdotV = saturate(NdotV);

    float lambdaV = NdotL * sqrt(alpha2 + (1 - alpha2) * NdotV * NdotV);
    float lambdaL = NdotV * sqrt(alpha2 + (1 - alpha2) * NdotL * NdotL);

    return 0.5f / max(lambdaV + lambdaL, 1e-7);
}

float4 main(VertexOutput inVert) : SV_TARGET0
{
    float depth = gDepth.Sample(gDepthSampler, inVert.uv).r;
    float3 fragWorldPosition = reconstructWorldPosition(depth, inVert.uv);
    float3 v = normalize(cameraData.position - fragWorldPosition);
    
    // If the fragment belongs to the background then the skybox will be sampled
    // else shading must be computed
    // TODO: add a separate skybox pass
    if (depth >= 1.0)
    {
        return skybox.Sample(samplers[1], -v);
    }
    
    // Light indipendent values
    float3 baseColor = gBaseColor.Sample(gBaseColorSampler, inVert.uv).rgb;
    float4 rawMaterialInfo = gMaterialInfo.Sample(gMaterialInfoSampler, inVert.uv);
    float roughness = rawMaterialInfo.g;
    float metalness = rawMaterialInfo.b;
    float3 n = normalize(gNormal.Sample(gNormalSampler, inVert.uv).rgb);
    float nDotV = max(dot(n, v), 0.0);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metalness);
    
    // TODO: remove when temporary indirect lighting solution is replaced
    float3 r = reflect(-v, n);
    
    // Iterating through lights
    // TODO: handle different light types
    float3 outColor = float3(0.0, 0.0, 0.0);
    for (uint i = 0; i < lightsUBO.lightsCount; i++)
    {
        LightData lightData = lightsUBO.lights[i];
        float3 l = -normalize(lightData.direction.xyz);
        float3 h = normalize(l + v);
        float nDotL = max(dot(l, n), 0.0);
        float hDotV = max(dot(h, v), 0.0);
        float nDotH = max(dot(n, h), 0.0);
        
        // Direct lighting evaluation
        float3 fresnel = F(f0, hDotV);
        float3 specular = fresnel * D(alphaSquared, nDotH) * G(alphaSquared, nDotL, nDotV);
        float3 diffuse = (float3(1.0, 1.0, 1.0) - fresnel) * diffuseBRDF(baseColor, metalness);
        float3 combinedBRDF = diffuse + specular;
        float3 directLighting = (lightData.color.rgb * lightData.intensity) * combinedBRDF * nDotL;
        
        // Adding i-th light contribution
        outColor += directLighting;
    }
    
    // Indirect lighting evaluation (not physically correct, it will be replaced once GI is implemented)
    //float3 kd = (1.0 - metalness);
    //float3 ambientDiffuse = 0.3 * baseColor * kd;
    //float3 specularIBL = skybox.Sample(samplers[1], r).rgb * fresnel * (1 - roughness * 0.7);
    //float3 indirectLighting = ambientDiffuse + specularIBL;
    //outColor += indirectLighting;
    
    // Tone mapping
    // TODO: find a suitable tone mapper
    float exposure = 0.005; // tweak
    outColor = 1.0 - exp(-outColor * exposure);
    
    return float4(outColor, 1.0);
}