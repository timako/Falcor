import Scene.Raster;

cbuffer PerFrameCB
{
    float4 gColor;
};

cbuffer MatBuffer
{
    float4x4 VPMat;
}

struct GSOut
{
    float3 posW : POSW;
    float4 posH : SV_POSITION;
    float3 normalW : NORMAL;
    float4 tangentW : TANGENT;
    float2 texC : TEXCRD;
    float4 prevPosH : PREVPOSH;
    nointerpolation GeometryInstanceID instanceID : INSTANCE_ID;
    nointerpolation uint materialID : MATERIAL_ID;
};



VSOut vsMain(VSIn vIn)
{
    StaticVertexData data = vIn.unpack();

    VSOut vOut;
    vOut.normalW = data.normal;
    const GeometryInstanceID instanceID = { vIn.instanceID };
    return defaultVS(vIn);
}

// [maxvertexcount(3)]
// void gsMain(triangle VSOut input[3], inout TriangleStream<GSOut> outStream)
// {
//     GSOut output;

//     for (int i = 0; i < 3; ++i)
//     {
//         output.posH = input[i].posH;
//         output.normalW = input[i].normalW;
//         output.tangentW = input[i].tangentW;
//         output.texC = input[i].texC;
//         output.posW = input[i].posW;
//         output.prevPosH = input[i].prevPosH;
//         output.instanceID = input[i].instanceID;
//         output.materialID = input[i].materialID;

//         outStream.Append(output);
//     }

//     outStream.RestartStrip();
// }

[maxvertexcount(2)]
void gsMain(line VSOut input[2], inout LineStream<GSOut> outStream)
{
    // 直接传递两个顶点到输出流，不做任何修改
    GSOut output;

    // 处理第一个顶点
    for (int i = 0; i < 2; ++i)
    {
        output.posH = input[i].posH;
        output.normalW = input[i].normalW;
        output.tangentW = input[i].tangentW;
        output.texC = input[i].texC;
        output.posW = input[i].posW;
        output.prevPosH = input[i].prevPosH;
        output.instanceID = input[i].instanceID;
        output.materialID = input[i].materialID;

        outStream.Append(output);
    }

    // 因为输入是线段，所以直接重新启动线段
    outStream.RestartStrip();
}

float4 psMain(GSOut psIn) : SV_Target
{
    return float4(1.0);
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
