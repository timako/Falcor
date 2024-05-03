import Scene.Raster;

// cbuffer PerFrameCB
// {
//     float4 gColor;
// };

cbuffer PerFrameCB
{
    float4x4 ShadowVP;
    float3 lightPos;
    float4 _DeepShadowLayerSplit;
};

Texture2D<float4> gShadowmap;
RWTexture2D<float4> gDOMtexture;

VSOut defaultVSVithVP(VSIn vIn, float4x4 VP)
{
    VSOut vOut;
    const GeometryInstanceID instanceID = { vIn.instanceID };

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(vIn.pos, 1.f)).xyz;
    vOut.posW = posW;
    vOut.posH = mul(VP, float4(posW, 1.f));

    vOut.instanceID = instanceID;
    vOut.materialID = gScene.getMaterialID(instanceID);

    vOut.texC = vIn.texC;
    vOut.normalW = mul(gScene.getInverseTransposeWorldMatrix(instanceID), vIn.unpack().normal);
    float4 tangent = vIn.unpack().tangent;
    vOut.tangentW = float4(mul((float3x3)gScene.getWorldMatrix(instanceID), tangent.xyz), tangent.w);

    // Compute the vertex position in the previous frame.
    float3 prevPos = vIn.pos;
    GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    if (instance.isDynamic())
    {
        uint prevVertexIndex = gScene.meshes[instance.geometryID].prevVbOffset + vIn.vertexID;
        prevPos = gScene.prevVertices[prevVertexIndex].position;
    }
    float3 prevPosW = mul(gScene.getPrevWorldMatrix(instanceID), float4(prevPos, 1.f)).xyz;
    vOut.prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));

    return vOut;
}

VSOut vsMain(VSIn vIn)
{
    return defaultVSVithVP(vIn, ShadowVP);
}

static float _Transmittance = 0.5;
static float _Alpha = 0.5;

float GetDomDistanceToFrontDepth(float FrontDepth, float LightSpaceZ, float bias)
{
#if UNITY_REVERSED_Z
    return max(0.0f, FrontDepth - LightSpaceZ - bias);
#else
    return max(0.0f, LightSpaceZ - FrontDepth - bias);
#endif
}

float ComputeTransmittanceFromAlpha(float alpha)
{
    float transmittance = (alpha * _Transmittance);
    return transmittance;
}

float4 ComputeAdditonalLightDeepLayerColor(int2 ipos, float3 positionWS, float transmittance)
{
    float width, height;
    gDOMtexture.GetDimensions(width, height);

    float4 positionLS = mul(ShadowVP, float4(positionWS, 1.0));

    float z = positionLS.z / positionLS.w;
    float2 positionSS = ipos * float2(width, height);

    float surfaceDepth = gShadowmap[ipos].x;

    // float surfaceDepth = SAMPLE_TEXTURE2D_ARRAY_LOD( _AdditionalLightDeepShadowDepth, sampler_AdditionalLightDeepShadowDepth, positionSS, _AdditionalLightDeepShadowDepth_Index, 0 ).x;

    float distanceToForntDepth = GetDomDistanceToFrontDepth(surfaceDepth, z, 0.0);
    float4 layerColor = gDOMtexture[ipos];

    layerColor.xyzw = distanceToForntDepth < _DeepShadowLayerSplit.wzyx; // 1,1,0,0

    layerColor.xyzw *= transmittance;      // alpha,alpha,0,0
    layerColor.xyzw = 1 - layerColor.xyzw; // 1-alpha,1-alpha,1,1
    gDOMtexture[ipos] = layerColor;
    return layerColor;
}

// float4 psMain(VSOut vIn) : SV_Target
// {
//     float3 posH = vIn.posH.xyz;    // 提取xyz坐标
//     posH.xy = posH.xy * 0.5 + 0.5; // 将x和y从[-1, 1]映射到[0, 1]
//     posH.z = saturate(posH.z);     // 确保z值在[0, 1]范围内（通常不必要，但为了安全）
//     posH /= 255.0;
//     return float4(posH, 1.0); // 使用修改后的posH作为颜色输出
// }

float4 psMain(VSOut vIn) : SV_Target
{
    float2 uv = vIn.texC;
    // float4 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, uv);

    float t = ComputeTransmittanceFromAlpha(_Alpha);
    int2 ipos = int2(vIn.posH.xy);

    return ComputeAdditonalLightDeepLayerColor(ipos, vIn.posW, t) / 25.0f;
}
