import common;
import nn;
import compiled;

#define M_PI 3.14159265853

struct VsOut
{
    float2 uv_screen : TEXCOORD;
    float4 position : SV_POSITION;
};

SamplerState LinearSampler;

Texture2D<float4> Position;
Texture2D<float4> Normal;
Texture2D<float4> Tangent;
Texture2D<float2> TexCoord;

float2 vector_to_eular(float3 vec) {
    vec = normalize(vec);
    float2 eular;
    eular.x = atan2(vec.y, vec.x);
    eular.y = acos(vec.z);
    return eular;
}


void main(VsOut IN, out float4 target_nn: SV_Target0, out float4 target_reference: SV_Target1) {
    float2 uv_screen = IN.uv_screen;
    float3 normal = Normal.Sample(LinearSampler, uv_screen).xyz;
    if (length(normal) < 0.1) {
        target_nn = float4(0, 0, 0, 1);
        target_reference = float4(0, 0, 0, 1);
        return;
    }

    float3 light_direction_world = iLights[0].dirW;
    float3 light_intensity = iLights[0].intensity;

    float3 tangent = Tangent.Sample(LinearSampler, uv_screen).xyz;
    float3 bitangent = cross(normal, tangent);
    float3 position = Position.Sample(LinearSampler, uv_screen).xyz;
    float2 uv_object = TexCoord.Sample(LinearSampler, uv_screen);

    // Gram－Schmidt
    normal = normalize(normal);
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    bitangent = normalize(bitangent - dot(bitangent, normal) * normal - dot(bitangent, tangent) * tangent);

    float3x3 TBN = float3x3(tangent, bitangent, normal);

    float3 wi = mul(TBN, -light_direction_world);
    float3 wo = mul(TBN, normalize(iCameraPosition - position));

    target_reference = float4(evaluate_legacy_brdf(uv_object, wi, wo) * light_intensity, 1);
    target_nn = float4(evaluate_neural_brdf(uv_object, wi, wo) * light_intensity, 1);
}