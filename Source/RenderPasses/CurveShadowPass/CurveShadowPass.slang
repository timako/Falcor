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

RWTexture2D<uint4> gVBuffer;
RWTexture2D<uint4> gVBuffer2;
RWTexture2D<uint4> gVBuffer3;
RWTexture2D<uint4> gVBuffer4;

RWTexture2D<uint> gHitCount;
RWTexture2D<float> gDepth;
RWTexture2D<float2> gMotionVector;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;
RWTexture2D<float> gMask;



#define is_valid_gDepth 1
#define is_valid_gMotionVector 1
#define is_valid_gViewW 1
#define is_valid_gTime 1
#define is_valid_gMask 1

#define is_valid(name) (is_valid_##name != 0)



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

    static const float4 colors[8] = {
        float4(0.0, 0.0, 0.0, 1.0), // 黑色
        float4(1.0, 0.0, 0.0, 1.0), // 红色
        float4(0.0, 1.0, 0.0, 1.0), // 绿色
        float4(0.0, 0.0, 1.0, 1.0), // 蓝色
        float4(1.0, 1.0, 0.0, 1.0), // 黄色
        float4(1.0, 0.0, 1.0, 1.0), // 紫色
        float4(0.0, 1.0, 1.0, 1.0), // 青色
        float4(1.0, 1.0, 1.0, 1.0)  //白色
    };

    uint pack(uint a, uint b, uint c, uint d) {
        return (a | (b << 4) | (c << 8) | (d << 12));
    }

    void unpack(uint packed, out uint a, out uint b, out uint c, out uint d) {
        a = packed & 0xF;
        b = (packed >> 4) & 0xF;
        c = (packed >> 8) & 0xF;
        d = (packed >> 12) & 0xF;
    }
    void addCount(inout uint original, uint channel, uint num) {
        uint a, b, c, d;
        unpack(original, a, b, c, d);
        if (channel == 0) {
            a += num;
        } else if (channel == 1) {
            b += num;
        } else if (channel == 2) {
            c += num;
        } else if (channel == 3) {
            d += num;
        }
        original = pack(a, b, c, d);
    }

    bool areEqual(uint4 original, uint4 num) {
        // 比较两个 uint4 向量的所有分量
        bool4 comparison = (original == num);

        // 检查所有分量是否都为真
        return comparison.x && comparison.y && comparison.z && comparison.w;
    }

    bool trywrite(inout uint4 original, uint4 num) {

        if(areEqual(original, uint4(0, 0, 0, 0))) {
            original = num;
            return true;
        } else {
            return false;
        }
    }

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

    bool tryUpdateChannel(inout uint packedData, uint primitiveIndex, uint2 pixel, uint channel) {
        if (packedData == 0) {
            packedData = packData(primitiveIndex, 1);
            addCount(gHitCount[pixel], channel, 1);
            return true;
        } else {
            uint count;
            uint primID = unpackData(packedData, count);
            if (primID == primitiveIndex) {
                packedData = packData(primitiveIndex, count + 1);
                addCount(gHitCount[pixel], channel, 1);
                return true;
            }
        }
        return false;
    }

    bool tryUpdateBuffer(inout uint4 packedData, uint4 data, uint2 pixel, uint channel) {
        if (areEqual(packedData, uint4(0, 0, 0, 0))) {
            packedData = data;
            addCount(gHitCount[pixel], channel, 1);
            return true;
        }
        else {
            uint count;
            uint4 primID = packedData;
            if (areEqual(primID, data)) {
                addCount(gHitCount[pixel], channel, 1);
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

            // if (!tryUpdateChannel(gVBuffer[pixel].x, primitiveIndex, pixel, 0)) {
            //     if (!tryUpdateChannel(gVBuffer[pixel].y, primitiveIndex, pixel, 1)) {
            //         if (!tryUpdateChannel(gVBuffer[pixel].z, primitiveIndex, pixel, 2)) {
            //             tryUpdateChannel(gVBuffer[pixel].w, primitiveIndex, pixel, 3);
            //         }
            //     }
            // }
            // gHitCount[pixel] = 1u;



            // uint4 data = hit.getData();
            // uint debugcount = 0;
            // if (!tryWriteBuffer(gVBuffer[pixel], data, pixel, 0)) {
            //     if (!tryWriteBuffer(gVBuffer2[pixel], data, pixel, 1)) {
            //         if (!tryWriteBuffer(gVBuffer3[pixel], data, pixel, 2)) {
            //             tryWriteBuffer(gVBuffer4[pixel], data, pixel, 3);
            //         }
            //     }
            // }
            // uint4 data = hit.getData();

            // uint4 data = uint4(primitiveIndex, primitiveIndex, primitiveIndex, 1);

            uint4 data = hit.getData();
            if (!tryUpdateBuffer(gVBuffer[pixel], data, pixel, 0)) {
                if (!tryUpdateBuffer(gVBuffer2[pixel], data, pixel, 1)) {
                    if (!tryUpdateBuffer(gVBuffer3[pixel], data, pixel, 2)) {
                        tryUpdateBuffer(gVBuffer4[pixel], data, pixel, 3);
                    }
                }
            }



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

    }

    void writeMiss(uint2 pixel, float3 rayOrigin, float3 rayDir)
    {
        gVBuffer[pixel] = {};

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
        // if (is_valid(gViewW))
        //     gViewW[pixel] = float4(-ray.dir, 0.f);

        uint count = 0;
        // if (gVBuffer[pixel].x != 0) count++;
        // if (gVBuffer[pixel].y != 0) count++;
        // if (gVBuffer[pixel].z != 0) count++;
        // if (gVBuffer[pixel].w != 0) count++;
        // gViewW[pixel] = colors[count];

        count += gHitCount[pixel] & 0xF;
        count += (gHitCount[pixel] >> 4) & 0xF;
        count += (gHitCount[pixel] >> 8) & 0xF;
        count += (gHitCount[pixel] >> 12) & 0xF;

        gViewW[pixel] = colors[count];

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
};
