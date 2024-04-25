__exported import Scene.Shading;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Scene.Raster;
import Utils.Math.MathHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGenerator;

// RWTexture2D<float> gShadowMap;
// RWTexture2D<float4> gViewW;
// RWTexture2D<float4> gDOMlayers;

struct DOMData {
    float4 layerdepths;
    float3 lightposW;
    float4x4 lightVP;
} ;

ConstantBuffer<DOMData> DOM;

struct DOMVSOut
{
    float2 texC : TEXCRD;

    // Per-triangle data
    nointerpolation GeometryInstanceID instanceID : INSTANCE_ID;
    nointerpolation uint materialID : MATERIAL_ID;

    float4 posH : SV_POSITION; ///< clip space

};

struct DOMGSOut
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD;
};

struct DOMPSOut
{
    float4 DOMresult : SV_TARGET0;
};

DOMVSOut vsMain(VSIn vsIn) {
    DOMVSOut vsOut;
    const GeometryInstanceID instanceID = { vsIn.instanceID };

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float4 posW = mul(worldMat, float4(vsIn.pos, 1.f));

    // 使用光源的视图投影矩阵进行位置变换
    vsOut.posH = mul(DOM.lightVP, posW);

    vsOut.texC = vsIn.texC;
    vsOut.instanceID = instanceID;
    vsOut.materialID = gScene.getMaterialID(instanceID);
    return vsOut;
}

// [maxvertexcount(4)]
// void gsMain(line DOMVSOut input[2], inout TriangleStream<DOMGSOut> outStream)
// {
//     // 计算方向
//     float3 dir = normalize(input[1].posH.xyz - input[0].posH.xyz);

//     // 计算相机到第一个顶点的方向
//     float3 cam2vertdir = normalize(input[0].posH.xyz - float3(0,0,0));

//     // 计算侧面方向
//     float3 right = normalize(cross(dir, cam2vertdir));

//     float radius = 0.05; // 暂时

//     // 构造四个顶点
//     float3 topLeft = input[0].posH.xyz - right * radius;
//     float3 topRight = input[0].posH.xyz + right * radius;
//     float3 bottomLeft = input[1].posH.xyz - right * radius;
//     float3 bottomRight = input[1].posH.xyz + right * radius;

//     // 输出顶点数据，设置UV等
//     DOMGSOut output;
//     output.position = float4(topLeft, 1.0);
//     output.uv = float2(0, 0);
//     outStream.Append(output);
//     output.position = float4(topRight, 1.0);
//     output.uv = float2(1, 0);
//     outStream.Append(output);
//     output.position = float4(bottomRight, 1.0);
//     output.uv = float2(1, 1);
//     outStream.Append(output);
//     output.position = float4(bottomLeft, 1.0);
//     output.uv = float2(0, 1);
//     outStream.Append(output);

//     // 完成四边形的绘制
//     outStream.RestartStrip();
// }

[maxvertexcount(2)]
void gsMain(line DOMVSOut input[2], inout LineStream<DOMGSOut> outStream)
{
    // 直接传递两个顶点到输出流，不做任何修改
    DOMGSOut output;

    // 处理第一个顶点
    output.position = input[0].posH; // 保持裁剪空间位置
    output.uv = input[0].texC;       // 保持纹理坐标
    outStream.Append(output);

    // 处理第二个顶点
    output.position = input[1].posH; // 保持裁剪空间位置
    output.uv = input[1].texC;       // 保持纹理坐标
    outStream.Append(output);

    // 因为输入是线段，所以直接重新启动线段
    outStream.RestartStrip();
}

DOMPSOut psMain(DOMGSOut gsOut) {
    DOMPSOut psOut;

    psOut.DOMresult = float4(0, 1, 0, 1); // 绿色

    return psOut;
}





