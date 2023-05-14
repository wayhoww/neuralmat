#include "Utils/Math/MathConstants.slangh"
import common;

SamplerState DefaultSampler;
Texture2D<float4> TextureAlbedo;

struct Parameters {
    float3 albedo;
};

float3 evaluate_brdf(float3 dir_i, float3 dir_o, Parameters params) {
    return params.albedo * dir_i.z / M_PI;
}
