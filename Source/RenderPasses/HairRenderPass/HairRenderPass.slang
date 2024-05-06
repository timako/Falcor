/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
__exported import Scene.Shading;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

RWTexture2D<float4> gVBuffer;
RWTexture2D<float> gDepth;
RWTexture2D<float2> gMotionVector;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;
RWTexture2D<float> gMask;
Texture2D<float3> gNp_tex;

Texture2D<float4> gDOMmap;
Texture2D<uint4> gVBuffermap;

#define is_valid_gDepth 1
#define is_valid_gMotionVector 1
#define is_valid_gViewW 1
#define is_valid_gTime 1
#define is_valid_gMask 1

#define is_valid(name) (is_valid_##name != 0)

struct HairRenderAsset {
    uint2 frameDim;
    uint frameCount;
    float4x4 ShadowVP;
    float3 lightPos;

    static const float4 colors[5] = {
        float4(0.0, 0.0, 0.0, 1.0), // 黑色
        float4(0.0, 1.0, 0.0, 1.0), // 绿色
        float4(0.0, 0.0, 1.0, 1.0), // 蓝色
        float4(1.0, 1.0, 0.0, 1.0), // 黄色
        float4(1.0, 0.0, 1.0, 1.0)  // 紫色
    };

    uint unpackData(uint data, out uint count) {
        count = (data >> 29) & 0x7;
        return data & 0x1FFFFFFF;
    }
    void beginTime(inout GpuTimer timer)
    {
        if (is_valid(gTime))
            timer.start();
    }

    void endTime(uint2 pixel, inout GpuTimer timer)
    {
        if (is_valid(gTime))
            gTime[pixel] = timer.getElapsed();
    }

    Ray generateRay(uint2 pixel)
    {
        return gScene.camera.computeRayPinhole(pixel, frameDim);
    }

    void visualizeHair(uint2 pixel) {
        uint type = 0;
        uint count;
        if (gVBuffermap[pixel].x != 0) {
            type += 1;
        }
        if (gVBuffermap[pixel].y != 0) {
            type += 1;
        }
        if (gVBuffermap[pixel].z != 0) {
            type += 1;
        }
        if (gVBuffermap[pixel].w != 0) {
            type += 1;
        }
        gVBuffer[pixel] = colors[type];
    }

    void RenderHair(uint2 pixel)
    {
        uint prims[4] = { 0, 0, 0, 0 };
        uint counts[4] = { 0, 0, 0, 0 };
        uint count;
        if (gVBuffermap[pixel].x != 0) {
            prims[0] = unpackData(gVBuffermap[pixel].x, count);
            counts[0] = count;
        }
        if (gVBuffermap[pixel].y != 0) {
            prims[1] = unpackData(gVBuffermap[pixel].y, count);
            counts[1] = count;
        }
        if (gVBuffermap[pixel].z != 0) {
            prims[2] = unpackData(gVBuffermap[pixel].z, count);
            counts[2] = count;
        }
        if (gVBuffermap[pixel].w != 0) {
            prims[3] = unpackData(gVBuffermap[pixel].w, count);
            counts[3] = count;
        }
        uint total = counts[0] + counts[1] + counts[2] + counts[3];
        
        Ray ray = generateRay(pixel);


    }
}
struct VBufferRT
{
    static const bool kComputeDepthOfField = true;
    static const bool kUseAlphaTest = true;
    static const uint kRayFlags = false;

    uint2 frameDim;
    uint frameCount;
    float4x4 ShadowVP;
    float3 lightPos;
    Camera lightCamera;

    Ray generateRay(uint2 pixel)
    {
        return gScene.camera.computeRayPinhole(pixel, frameDim);
    }

    Ray generateRay8xMSAA(uint2 pixel);
    {
        return gScene.camera.computeRayPinhole(pixel, frameDim);
    }

    Ray generateRayWithLightVP(uint2 pixel)
    {
        return gScene.camera.computeRayPinholeWithLightVP(pixel, frameDim, lightPos, ShadowVP);
    }

    static const float4 colors[4] = {
        float4(1.0, 0.0, 0.0, 1.0), // 红色
        float4(0.0, 1.0, 0.0, 1.0), // 绿色
        float4(0.0, 0.0, 1.0, 1.0), // 蓝色
        float4(1.0, 1.0, 0.0, 1.0)  // 黄色
    };

    uint packData(uint number, uint count) {
        if (count >= 8) {
            return 0;
        }
        return (count << 29) | (number & 0x1FFFFFFF); // 将count放到高3位，number放到低29位
    }

    uint unpackData(uint data, out uint count) {
        count = (data >> 29) & 0x7;
        return data & 0x1FFFFFFF;
    }

    bool tryUpdateChannel(inout uint packedData, uint primitiveIndex) {
        if (packedData == 0) {
            packedData = packData(primitiveIndex, 1);
            return true;
        } else {
            uint count;
            uint primID = unpackData(packedData, count);
            if (primID == primitiveIndex) {
                packedData = packData(primitiveIndex, count + 1);
                return true;
            }
        }
        return false;
    }

    void writeHit(uint2 pixel, float3 rayOrigin, float3 rayDir, const HitInfo hit)
    {
        // gVBuffer[pixel] = hit.getData();

        VertexData v;
        float depth = 1.f;
        float3 prevPosW = float3(0.f);
        float2 motionVector = float2(0.f);

        if (hit.getType() == HitType::Triangle)
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            GeometryInstanceID instanceID = triangleHit.instanceID;
            uint primitiveIndex = triangleHit.primitiveIndex;

            // if (!tryUpdateChannel(gVBuffer[pixel].x, primitiveIndex)) {
            //     if (!tryUpdateChannel(gVBuffer[pixel].y, primitiveIndex)) {
            //         if (!tryUpdateChannel(gVBuffer[pixel].z, primitiveIndex)) {
            //             tryUpdateChannel(gVBuffer[pixel].w, primitiveIndex);
            //         }
            //     }
            // }

            v = gScene.getVertexData(triangleHit);
            prevPosW = gScene.getPrevPosW(triangleHit);
        }
        else if (hit.getType() == HitType::DisplacedTriangle)
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            v = gScene.getVertexData(displacedTriangleHit, -rayDir);
            prevPosW = gScene.getPrevPosW(displacedTriangleHit);
        }
        else if (hit.getType() == HitType::Curve)
        {
            const CurveHit curveHit = hit.getCurveHit();
            v = gScene.getVertexDataFromCurve(curveHit);
            prevPosW = gScene.getPrevPosWFromCurve(curveHit);
        }
        else if (hit.getType() == HitType::SDFGrid)
        {
            const SDFGridHit sdfGridHit = hit.getSDFGridHit();
            v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);

            prevPosW = gScene.getPrevPosWFromSDFGrid(sdfGridHit, v.posW);
        }

        // if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle || hit.getType() == HitType::Curve ||
        //     hit.getType() == HitType::SDFGrid)
        // {
        //     // Compute depth similar to raster (NDC).
        //     float4 curPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(v.posW, 1.f));
        //     depth = curPosH.z / curPosH.w;

        //     // Compute motion vector.
        //     float2 pixelPos = pixel + float2(0.5f, 0.5f);
        //     float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
        //     // Remove camera jitter from motion vector
        //     motionVector = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
        // }

        // if (is_valid(gDepth))
        //     gDepth[pixel] = depth;
        // if (is_valid(gMotionVector))
        //     gMotionVector[pixel] = motionVector;
        // if (is_valid(gMask))
        //     gMask[pixel] = 1.0f;
    }

    void writeMiss(uint2 pixel, float3 rayOrigin, float3 rayDir)
    {
        // gVBuffer[pixel] = {};

        if (is_valid(gDepth))
            gDepth[pixel] = 1.f;
        if (is_valid(gMotionVector))
            gMotionVector[pixel] = {};
        if (is_valid(gMask))
            gMask[pixel] = 0.0f;
    }

    void writeAux(uint2 pixel, const Ray ray)
    {
        // Write view direction.
        if (is_valid(gViewW))
            gViewW[pixel] = float4(-ray.dir, 0.f);
    }


};
