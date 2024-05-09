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
#include "Utils/Math/MathConstants.slangh"
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

RWTexture2D<uint> gHitCount;
RWTexture2D<float4> gDOMmap;
RWTexture2D<uint4> gVBuffermap;
RWTexture2D<uint4> gVBuffermap2;
RWTexture2D<uint4> gVBuffermap3;
RWTexture2D<uint4> gVBuffermap4;

#define is_valid_gDepth 1
#define is_valid_gMotionVector 1
#define is_valid_gViewW 1
#define is_valid_gTime 1
#define is_valid_gMask 1

#define is_valid(name) (is_valid_##name != 0)

inline float gaussian_function(float b, float t)
{
    return clamp(exp(-t * t / (2 * b * b)) / sqrt(2.0f * M_PI) / b, 0.0001f, 1000.0f);
}

inline float degToRad(float value) { return value * (M_PI / 180.0f); }

#define INTEGRATOR_NUM_SAMPLE 140
#define AZIMUTHAL_PRECOMPUTE_RESOLUTION 64
#define GAUSSIAN_DETECTOR_SAMPLES 2048

struct HairRenderAsset {
    uint2 frameDim;
    uint frameCount;
    float4x4 ShadowVP;
    float3 lightPos;
    static const float3 lightColor = float3(0.7216, 0.5255, 0.0431);
    static const float lightintensity = 1.5;

    float3 albedo;
    static const float m_eta = 1.55f;
    // longtitudinal shift
    static const float m_alpha = degToRad(-5.0f);
    static const float alpha[3] = { m_alpha,
                                    -m_alpha / 2,
                                    -3 * m_alpha / 2};
    // longtitudinal width
    static const float m_beta = degToRad(3.f);
    static const float beta[3] = { m_beta,
                                    m_beta / 2,
                                    m_beta * 2 };
    static const float3 m_absorption = float3(0.3, 0.6, 0.9f);

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

    bool areEqual(uint4 original, uint4 num) {
        // 比较两个 uint4 向量的所有分量
        bool4 comparison = (original == num);

        // 检查所有分量是否都为真
        return comparison.x && comparison.y && comparison.z && comparison.w;
    }

    uint pack(uint a, uint b, uint c, uint d) {
        return (a | (b << 4) | (c << 8) | (d << 12));
    }

    void unpack(uint packed, out uint a, out uint b, out uint c, out uint d) {
        a = packed & 0xF;
        b = (packed >> 4) & 0xF;
        c = (packed >> 8) & 0xF;
        d = (packed >> 12) & 0xF;
    }

    float3 debugColor(uint id) {
        return colors[id % 5].xyz;
    }

    float3 debugColor(float data) {
        return float3(data);
    }


    float3 approxNp(float cos_theta_d, float phi, int i)
    {
        // costhetad -> phi
        // float cos_theta_d = std::clamp(float(y) / float(AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1) + float(0.00001), 0.0f, 1.0f);
        // float phi = M_PI * 2.f * x / (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1.f);

        float u = (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1) * phi * M_1_PI * 0.5f;
        float v = (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1) * cos_theta_d;
        int x0 = clamp(int(u), 0, AZIMUTHAL_PRECOMPUTE_RESOLUTION * 3 - 2);
        int y0 = clamp(int(v), 0, AZIMUTHAL_PRECOMPUTE_RESOLUTION - 2);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        u = clamp(u - x0, 0.0f, 1.0f);
        v = clamp(v - y0, 0.0f, 1.0f);

        // return clamp( (gNp_tex[x0 * 3 + i][y0] * (1.0f - u) + gNp_tex[x1 * 3 + i][y0] * u) * (1.0f - v), float3(0.0001f), float3(1.0f) ) + clamp((gNp_tex[x0 * 3 + i][y1] * (1.0f - u) + gNp_tex[x1 * 3 + i][y1] * u) * v, float3(0.0001f), float3(1.0f));
        return clamp( (gNp_tex[uint2(x0 * 3 + i, y0)] * (1.0f - u) + gNp_tex[uint2(x1 * 3 + i, y0)] * u) * (1.0f - v), float3(0.0001f), float3(1.0f) ) + clamp((gNp_tex[uint2(x0 * 3 + i, y1)] * (1.0f - u) + gNp_tex[uint2(x1 * 3 + i, y1)] * u) * v, float3(0.0001f), float3(1.0f));
    }

    float3 eval(const float3 wi, const float3 wo) {
        // if (min(wi.z, -wo.z) < kMinCosTheta)
        //     return float3(0.0f);

        float theta_i = asin(clamp(wi.x, -1.0f, 1.0f));
        float theta_r = asin(clamp(wo.x, -1.0f, 1.0f));

        float phi_i = atan2(wi.y, wi.z);
        float phi_r = atan2(wo.y, wo.z);
        float phi = phi_r - phi_i, theta_d = abs((theta_r - theta_i)) / 2, theta_h = (theta_i + theta_r) / 2;



        if (phi < 0.f)
            phi += M_PI * 2;
        if (phi > M_PI * 2)
            phi -= M_PI * 2;



        float3 result = float3(0.f);

        // M term, we'll simply use gaussian distribution here
        float Mp[3];
        for (int i = 0; i < 3; i++)
        {
            Mp[i] = gaussian_function(beta[i], theta_h - alpha[i]);
        }

        float cos_theta_d = cos(theta_d);

        Mp[0] = gaussian_function(beta[0], theta_h * 2);

        // result = debugColor(Mp[0]);

        // result = gNp_tex[uint2(1, 0)];

        result += approxNp(cos_theta_d, phi, 0);
        // result += Mp[1] * approxNp(cos_theta_d, phi, 1);
        // result += Mp[2] * approxNp(cos_theta_d, phi, 2);

        // result *= lightColor;

        return result;
    }

    float3 findAndNormalizeLongestSides(float3 v1, float3 v2, float3 v3) {

        // 计算三角形的三条边
        float3 edge1 = v2 - v1;
        float3 edge2 = v3 - v2;
        float3 edge3 = v1 - v3;

        // 找出最长的边
        float len1 = length(edge1);
        float len2 = length(edge2);
        float len3 = length(edge3);

        float3 longestEdge;

        if (len1 > len2 && len1 > len3) {
            longestEdge = edge1;
        } else if (len2 > len3) {
            longestEdge = edge2;
        } else {
            longestEdge = edge3;
        }

        // 归一化这个向量
        float3 normalizedEdge = normalize(longestEdge);
        return normalizedEdge;
    }

    void visualizeHair(uint2 pixel) {
        if (pixel.x <= AZIMUTHAL_PRECOMPUTE_RESOLUTION * 3 && pixel.y <= AZIMUTHAL_PRECOMPUTE_RESOLUTION)
            gViewW[pixel] = float4(gNp_tex[pixel], 1.0);
        uint4 datas[4] = { gVBuffermap[pixel],
                        gVBuffermap2[pixel],
                        gVBuffermap3[pixel],
                        gVBuffermap4[pixel] };

        HitInfo hits[4] = { HitInfo(gVBuffermap[pixel]) ,
                            HitInfo(gVBuffermap2[pixel]),
                            HitInfo(gVBuffermap3[pixel]),
                            HitInfo(gVBuffermap4[pixel]) };

        VertexData v[4] = { 0,
                            0,
                            0,
                            0 };

        TriangleHit triangleHit[4] = { 0,
                            0,
                            0,
                            0 };

        StaticVertexData vertices[4][3];

        for (uint i = 0; i < 4; i++) {
            if (!areEqual(datas[i], uint4(0, 0, 0, 0))) {
                if (hits[i].getType() == HitType::Triangle) {
                    triangleHit[i] = hits[i].getTriangleHit();
                    v[i] = gScene.getVertexData(triangleHit[i], vertices[i]);

                }
            }
            else {
                break;
            }
        }

        // float3 vertTangentW[4];
        // for (int i = 0; i < 4; i++) {
        //     if (areEqual(datas[i], uint4(0, 0, 0, 0))) {
        //         break;
        //     }
        //     else {
        //         vertTangentW[i] = findAndNormalizeLongestSides(vertices[i][0].position, vertices[i][1].position, vertices[i][2].position);
        //     }
        // }

        // GeometryInstanceID instanceID = triangleHit[0].instanceID;
        // uint primitiveIndex = triangleHit[0].primitiveIndex;
        // float3 barycoords = triangleHit[0].getBarycentricWeights();

        // float3 normalW = v[0].normalW;

        // float3 lightDir = normalize(v[0].posW - lightPos);
        // float3 lightDirVis = (lightDir + float3(1.0, 1.0, 1.0)) / 2.0;
        // float diff = dot(normalW, lightDir);
        // float3 diffuse = lightColor * diff * lightintensity;
        // gVBuffer[pixel] = float4(diffuse.x, diffuse.y, diffuse.z, 1.0);

        float3 color = float3(0.0, 0.0, 0.0);
        uint weights[4];
        unpack(gHitCount[pixel], weights[0], weights[1], weights[2], weights[3]);
        Ray ray = generateRay(pixel);
        float3 raydir = ray.dir;


        for (int i = 0; i < 4; i++) {
            if (areEqual(datas[i], uint4(0, 0, 0, 0))) {
            }
            else {
                float3 normalW = v[i].normalW;   ///< Shading normal in world space (normalized).
                float3 lightDir = -normalize(v[i].posW - lightPos);
                float lightLen = length(v[i].posW - lightPos);
                float4 tangentW = v[i].tangentW; ///< Shading tangent in world space (normalized). The last component is guaranteed to be +-1.0 or zero if tangents are missing.
                // float4 tangentW = float4(vertTangentW[i], 1.0);
                float3 raydir = -ray.dir;

                float3 bitangentW = cross(normalW, tangentW.xyz) * tangentW.w;

                float3x3 TBN = float3x3(tangentW.xyz, bitangentW, normalW); // 构建切线空间矩阵
                float3x3 TBN_inv = transpose(TBN);                          // 切线空间矩阵的逆（转置）

                // 转换 lightDir 和 raydir 到切线空间
                float3 lightDirT = mul(lightDir, TBN_inv);
                float3 rayDirT = mul(raydir, TBN_inv);

                // if(lightDirT.z < 0.0)
                //     lightDirT = -lightDirT;

                // if(rayDirT.z < 0.0)
                //     rayDirT = -rayDirT;

                float3 lightDirTVis = (lightDirT + 1.0) / 2;
                float3 rayDirTVis = (rayDirT + 1.0) / 2;

                // float diff = dot(normalW, lightDir);
                // float3 diffuse = lightColor * diff * lightintensity;
                // float3 diffuse = eval(lightDirT, rayDirT);
                float3 diffuse = eval(lightDirT, rayDirT);
                // float3 diffuse = v[i].tangentW.xyz;
                color += diffuse * weights[i];

            }
        }
        if (weights[0] + weights[1] + weights[2] + weights[3] != 0u)
            color /= (weights[0] + weights[1] + weights[2] + weights[3]) ;

        if (color.x == 0.0 && color.y == 0.0 && color.z == 0.0)
            color = gScene.envMap.eval(raydir);

        gVBuffer[pixel] = float4(color, 1.0);


    }


    // void RenderHair(uint2 pixel)
    // {


    // }
}
