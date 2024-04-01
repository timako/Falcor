import Scene.Raster;

cbuffer PerFrameCB
{
    float4 gColor;
};

cbuffer MatBuffer
{
    float4x4 VPMat;
}

VSOut vsMain(VSIn vIn)
{
    StaticVertexData data = vIn.unpack();

    VSOut vOut;
    vOut.normalW = data.normal;
    const GeometryInstanceID instanceID = { vIn.instanceID };
    return defaultVS(vIn);
}

float4 psMain(VSOut vIn) : SV_Target
{
    float3 normalColor = vIn.normalW * 0.5 + 0.5; // 将[-1, 1]范围的法线映射到[0, 1]
    return float4(normalColor, 1.0);             // Alpha设为1
}

// ShadowPassVSOut vsMain(VSIn vIn)
// {
//     ShadowPassVSOut vOut;
//     const GeometryInstanceID instanceID = { vIn.instanceID };
//     float4x4 worldMat = gScene.getWorldMatrix(instanceID);
//     vOut.pos = mul(worldMat, float4(vIn.pos, 1.f));
// #ifdef _APPLY_PROJECTION
//     vOut.pos = mul(gScene.camera.getViewProj(), vOut.pos);
// #endif

//     vOut.texC = vIn.texC;
//     return vOut;
// }
