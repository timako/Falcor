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
#pragma once
#include "HSPData.slang"
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

struct HSPData
{
    float4x4 globalMat;

    float depthBias = 0.005f;

    float3 lightPos;
    float lightBleedingReduction = 0;

    uint32_t padding;

#ifndef HOST_CODE
    Texture2DArray shadowMap;

#endif
};

class HairShadowPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HairShadowPass, "HairShadowPass", "Insert pass description here.");

    static ref<HairShadowPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<HairShadowPass>(pDevice, props);
    }

    HairShadowPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;

    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void setLight(ref<Light> pLight);
    void createShadowMatrix(const PointLight* pLight, const float3 center, float radius, float fboAspectRatio, float4x4& shadowVP);
    void createShadowMatrix(const Light* pLight, const float3& center, float radius, float fboAspectRatio, float4x4& shadowVP);
    void GenerateShadowPass(const Camera* pCamera, float aspect);
private:

    ref<Scene> mpScene;
    ref<Program> mpProgram;
    ref<GraphicsState> mpGraphicsState;
    ref<RasterizerState> mpRasterState;
    ref<ProgramVars> mpVars;
    ref<Fbo> mpFbo;

    ref<Light> mpLight;

    // Shadow-pass
    struct
    {
        ref<Fbo> pFbo;
        float fboAspectRatio;
        ref<Program> pProgram;
        ref<ProgramVars> pVars;
        ref<GraphicsState> pGraphicsState;
        ref<RasterizerState> pRasterState;
        float2 mapSize;

    } mShadowPass;

    HSPData mHSPData;

    float3 mLightPos;
    float4x4 mLightVP;
};
