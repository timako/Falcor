import Scene.RaytracingInline;
import HairRenderPass;

ConstantBuffer<VBufferRT> gVBufferRT;
ConstantBuffer <HairRenderAsset> gHairRenderAsset;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;

    if (any(pixel >= gVBufferRT.frameDim))
        return;

    GpuTimer timer;
    gHairRenderAsset.beginTime(timer);

    gHairRenderAsset.visualizeHair(pixel);


    // if (any(pixel >= gVBufferRT.frameDim))
    //     return;

    // GpuTimer timer;
    // gVBufferRT.beginTime(timer);

    // // const Ray ray = gVBufferRT.generateRay(pixel);

    // SceneRayQuery<VBufferRT::kUseAlphaTest> sceneRayQuery;
    // HitInfo hit;
    // float hitT;
    // uint hitcount = 0u;
    // for (int i = 0; i < 7; i++) {
    //     float2 jitter = Hammersley(i, 7) - float2(0.5f, 0.5f);
    //     Ray ray = gScene.camera.computeRayPinholeWithGivenJitter(pixel, gVBufferRT.frameDim, jitter);
    //     if (sceneRayQuery.traceRay(ray, hit, hitT, VBufferRT::kRayFlags, 0xff)) {
    //         gVBufferRT.writeHit(pixel, ray.origin, ray.dir, hit);
    //         hitcount++;
    //     }
    // }

    // Ray ray = gVBufferRT.generateRay(pixel);

    // if (hitcount == 0u) {
    //     gVBufferRT.writeMiss(pixel, ray.origin, ray.dir);
    // }

    // if (sceneRayQuery.traceRay(ray, hit, hitT, VBufferRT::kRayFlags, 0xff))
    // {
    //     gVBufferRT.writeHit(pixel, ray.origin, ray.dir, hit);
    // }
    // else
    // {
    //     gVBufferRT.writeMiss(pixel, ray.origin, ray.dir);
    // }



    // gVBufferRT.writeAux(pixel, ray);

    gHairRenderAsset.endTime(pixel, timer);
}
