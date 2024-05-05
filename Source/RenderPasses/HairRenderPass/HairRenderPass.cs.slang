import Scene.RaytracingInline;
import HairRenderPass;

ConstantBuffer<VBufferRT> gVBufferRT;

float RadicalInverse(uint bits) {
    // reverse bit
    // 高低16位换位置
    bits = (bits << 16u) | (bits >> 16u);
    // A是5的按位取反
    bits = ((bits & 0x55555555) << 1u) | ((bits & 0xAAAAAAAA) >> 1u);
    // C是3的按位取反
    bits = ((bits & 0x33333333) << 2u) | ((bits & 0xCCCCCCCC) >> 2u);
    bits = ((bits & 0x0F0F0F0F) << 4u) | ((bits & 0xF0F0F0F0) >> 4u);
    bits = ((bits & 0x00FF00FF) << 8u) | ((bits & 0xFF00FF00) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 Hammersley(uint i, uint N) {
    return float2(float(i) / float(N), RadicalInverse(i));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gVBufferRT.frameDim))
        return;

    GpuTimer timer;
    gVBufferRT.beginTime(timer);

    // const Ray ray = gVBufferRT.generateRay(pixel);

    SceneRayQuery<VBufferRT::kUseAlphaTest> sceneRayQuery;
    HitInfo hit;
    float hitT;
    uint hitcount = 0u;
    for (int i = 0; i < 7; i++) {
        float2 jitter = Hammersley(i, 7) - float2(0.5f, 0.5f);
        Ray ray = gScene.camera.computeRayPinholeWithGivenJitter(pixel, gVBufferRT.frameDim, jitter);
        if (sceneRayQuery.traceRay(ray, hit, hitT, VBufferRT::kRayFlags, 0xff)) {
            gVBufferRT.writeHit(pixel, ray.origin, ray.dir, hit);
            hitcount++;
        }
    }

    Ray ray = gVBufferRT.generateRay(pixel);

    if (hitcount == 0u) {
        gVBufferRT.writeMiss(pixel, ray.origin, ray.dir);
    }

    // if (sceneRayQuery.traceRay(ray, hit, hitT, VBufferRT::kRayFlags, 0xff))
    // {
    //     gVBufferRT.writeHit(pixel, ray.origin, ray.dir, hit);
    // }
    // else
    // {
    //     gVBufferRT.writeMiss(pixel, ray.origin, ray.dir);
    // }



    gVBufferRT.writeAux(pixel, ray);

    gVBufferRT.endTime(pixel, timer);
}
