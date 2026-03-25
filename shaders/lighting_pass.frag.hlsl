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
    float       skyboxIntensity;
    float       exposure;
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

// ACES film tonemapper (Narkowicz)
// ref. https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
float3 tonemapACES(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

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
    float3 outColor = float3(0.0, 0.0, 0.0);
    float depth = gDepth.Sample(gDepthSampler, inVert.uv).r;
    float3 fragWorldPosition = reconstructWorldPosition(depth, inVert.uv);
    float3 v = normalize(cameraData.position - fragWorldPosition);
    
    // If the fragment falls on the far plane
    // the skybox is sampled, else shading is performed
    // TODO: add a separate skybox pass
    if (depth >= 1.0)
    {
        outColor = skybox.Sample(samplers[1], -v);
        outColor = tonemapACES(cameraData.skyboxIntensity * outColor);
        return float4(outColor, 1.0);
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
    float3 r = reflect(-v, n);
    
    // Iterating through the lights (by guaranteeing that lights are in the same
    // order regardless of what fragment/thread is executing, branch divergence
    // shouldn't be a problem)
    for (uint i = 0; i < lightsUBO.lightsCount; i++)
    {
        LightData lightData = lightsUBO.lights[i];
        float3 l;
        float attenuation = 1.0;
        
        if (lightData.type == 0) // DIRECTIONAL
        {
            l = -normalize(lightData.direction.xyz);
        }
        else // POINT and SPOT (they share the same setup)
        {
            l = normalize(lightData.position.xyz - fragWorldPosition);
            float distance = length(lightData.position.xyz - fragWorldPosition);
            
            attenuation = 1.0 / max(distance * distance, 0.001);
            if(lightData.range > 0.0) // if 0.0 -> infinite range
            {
                // windowing function multiplied by the inverse-square attenuation (see Real-Time Rendering)
                attenuation = max(min(1.0 - pow(distance / lightData.range, 4), 1), 0) / (distance * distance);
            }
            
            if (lightData.type == 2) // SPOT
            {
                // Spotlight cone check
                // Checks how close the fragment is with respect to the spot light direction
                float spotAngle = dot(-l, normalize(lightData.direction.xyz));
                if (spotAngle > lightData.outerConeAngle)
                {
                    // Smooth falloff between inner and outer cone
                    float spotFactor = smoothstep(lightData.outerConeAngle, lightData.innerConeAngle, spotAngle);
                    attenuation *= spotFactor;
                }
                else
                {
                    // Outside the spotlight cone
                    attenuation = 0.0; 
                }
            }
        }
    
        float3 h = normalize(l + v);
        float nDotL = max(dot(l, n), 0.0);
        float hDotV = max(dot(h, v), 0.0);
        float nDotH = max(dot(n, h), 0.0);
        
        // Direct lighting evaluation
        float3 fresnel = F(f0, hDotV);
        float3 specular = fresnel * D(alphaSquared, nDotH) * G(alphaSquared, nDotL, nDotV);
        float3 diffuse = (float3(1.0, 1.0, 1.0) - fresnel) * diffuseBRDF(baseColor, metalness);
        float3 combinedBRDF = diffuse + specular;
        float3 directLighting = (lightData.color.rgb * lightData.intensity * attenuation) * combinedBRDF * nDotL;
        
        // Adding i-th light contribution
        outColor += directLighting;
    }
    
    // Tone mapping
    outColor = tonemapACES(outColor * cameraData.exposure);
    
    return float4(outColor, 1.0);
}