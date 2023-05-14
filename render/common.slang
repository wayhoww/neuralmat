import Scene.Lights.LightData;

cbuffer PerFrameCB
{
    uint3 iThreads;
    uint2 iResolution; // graphics pass only

    float iTime;
    float iDeltaTime;
    
    float2 iMousePosition;
    int2 iMouseCoordinate;
    int2 iMouseLastCoordinate;
    bool iMouseLeftButtonDown;

    float3 iCameraPosition;
    float3 iCameraU;
    float3 iCameraV;
    float3 iCameraW;

    uint2 iFrameDim;

    uint iLightCount;
};

StructuredBuffer<LightData> iLights;