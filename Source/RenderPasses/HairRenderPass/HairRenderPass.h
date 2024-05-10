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
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "HairMaterial_re.h"

using namespace Falcor;

class HairRenderPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HairRenderPass, "HairRenderPass", "Insert pass description here.");

    static ref<HairRenderPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<HairRenderPass>(pDevice, props);
    }

    HairRenderPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void updateFrameDim(const uint2 frameDim);
    void executeCompute(RenderContext* pRenderContext, const RenderData& renderData);
    void bindShaderData(const ShaderVar& var, const RenderData& renderData);
    void recreatePrograms();
    DefineList getShaderDefines(const RenderData& renderData);

    void setLight(ref<Light> pLight);
    void GenerateShadowPass(const Camera* pCamera, float aspect);
    void createShadowMatrix(const PointLight* pLight, const float3 center, float radius, float fboAspectRatio, float4x4& shadowVP);
    void createShadowMatrix(const Light* pLight, const float3& center, float radius, float fboAspectRatio, float4x4& shadowVP);

private:

    ref<Scene> mpScene;
    ref<Program> mpProgram;
    ref<GraphicsState> mpGraphicsState;
    ref<ProgramVars> mpVars;
    ref<Fbo> mpFbo;
    ref<Light> mpLight;

    ref<ComputePass> mpComputePass;

    uint2 mFrameDim = {};
    float2 mInvFrameDim = {};
    uint32_t mFrameCount = 0;
    float3 mLightPos;
    ResourceFormat mVBufferFormat = HitInfo::kDefaultFormat;
    RenderPassHelpers::IOSize mOutputSizeSelection = RenderPassHelpers::IOSize::Default;
    /// Output size in pixels when 'Fixed' size is selected.
    uint2 mFixedOutputSize = {512, 512};

    float4x4 mLightVP;

    ref<SampleGenerator> mpSampleGenerator;
    MarschnerHair mHairBSDF;

    ref<Texture> GD_tex;
    ref<Texture> Np_tex;

};
