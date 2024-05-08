import Scene.RaytracingInline;
import HairRenderPass;

ConstantBuffer<HairRenderAsset> gHairRenderAsset = {};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;

    if (any(pixel >= gHairRenderAsset.frameDim))
        return;

    GpuTimer timer;
    gHairRenderAsset.beginTime(timer);

    gHairRenderAsset.visualizeHair(pixel);

    gHairRenderAsset.endTime(pixel, timer);
}
