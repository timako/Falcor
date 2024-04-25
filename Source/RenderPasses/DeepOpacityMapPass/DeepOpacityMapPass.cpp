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
#include "DeepOpacityMapPass.h"
#include "Scene/HitInfo.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "RenderGraph/RenderPassHelpers.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DeepOpacityMapPass>();
}

void DeepOpacityMapPass::setLight(ref<Light> pLight)
{
    if(pLight){
        mpLight = pLight;
    }

}

void DeepOpacityMapPass::createShadowMatrix(const PointLight* pLight, const float3 center, float radius, float fboAspectRatio, float4x4& shadowVP){
    const float3 lightPos = pLight->getWorldPosition();
    mLightPos = lightPos;
    const float3 lookat = center; // 点光源定义direction 可以换成center ?
    const float3 Direction = math::normalize(center-lightPos);
    float3 up(0, 1, 0);
    if (abs(dot(up, Direction)) >= 0.95f)
    {
        up = float3(1, 0, 0); // 如果光源方向与up向量几乎平行（即它们的点积的绝对值接近1），则更改up向量为(1, 0, 0)，以避免计算时的奇异性。
    }

    float4x4 view = math::matrixFromLookAt(lightPos, lookat, up); // 从世界坐标系到光源视角坐标系的变换

    float distFromCenter = math::length(lightPos - center);
    float nearZ = std::max(0.1f, distFromCenter - radius);
    float maxZ = std::min(radius * 2, distFromCenter + radius);
    float angle = pLight->getOpeningAngle() * 2;

    float4x4 proj = math::perspective(angle, fboAspectRatio, nearZ, maxZ);

    shadowVP = math::mul(proj , view);

}

void DeepOpacityMapPass::createShadowMatrix(const Light* pLight, const float3& center, float radius, float fboAspectRatio, float4x4& shadowVP)
{
    if(pLight->getType() == LightType::Point){
        PointLight* ppLight = (PointLight*)pLight;
        ppLight->setOpeningAngle((float)6.28319 / (float)8.0); // 设置正常的视场角（45度）
        createShadowMatrix((PointLight*)pLight, center, radius, fboAspectRatio, shadowVP);
    }
}

void DeepOpacityMapPass::GenerateShadowPass(const Camera* pCamera, float aspect){
    AABB mpSceneBB = mpScene->getSceneBounds();
    const float3 center((mpSceneBB.minPoint.x + mpSceneBB.maxPoint.x)/2,
                        (mpSceneBB.minPoint.y + mpSceneBB.maxPoint.y)/2,
                        (mpSceneBB.minPoint.z + mpSceneBB.maxPoint.z)/2);

    const float radius = math::length(mpSceneBB.extent()) / 2;

    createShadowMatrix(mpLight.get(), center, radius, aspect, mLightVP);
}
/*
    struct ChannelDesc
    {
        std::string name;      ///< Render pass I/O pin name.
        std::string texname;   ///< Name of corresponding resource in the shader, or empty if it's not a shader variable.
        std::string desc;      ///< Human-readable description of the data.
        bool optional = false; ///< Set to true if the resource is optional.
        ResourceFormat format = ResourceFormat::Unknown; ///< Default format is 'Unknown', which means let the system decide.
    };
*/
namespace{
    const std::string kProgramFile = "RenderPasses/DeepOpacityMapPass/DeepOpacityMapPass.slang";
    const RasterizerState::CullMode kDefaultCullMode = RasterizerState::CullMode::None;
    const char kInputViewDir[] = "viewW";

    const ChannelList kInputChannels = {
        // clang-format off
        { "kshadowMap",        "gShadowMap",     "Shadow Map from hair",                            false /* optional */, ResourceFormat::R32Float    },
        { kInputViewDir,       "gViewW",         "World-space view direction (xyz float format)" ,  true /* optional */,  ResourceFormat::RGBA32Float },
        // clang-format on
    };

    const ChannelList kOutputChannels = {
        // clang-format off
        { "DOM layer pass",   "gDOMlayers",      "Output color (sum of direct and indirect)",       false,                ResourceFormat::RGBA32Float },
        // clang-format on
    };

    const std::string kVBufferName = "gDOMlayers";
    const std::string kVBufferDesc = "Output color (sum of direct and indirect)";
}

DeepOpacityMapPass::DeepOpacityMapPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    // Check for required features.
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("VBufferRaster requires Shader Model 6.2 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("VBufferRaster requires pixel shader barycentrics support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
        FALCOR_THROW("VBufferRaster requires rasterizer ordered views (ROVs) support.");


    // Initialize graphics state
    mRaster.pState = GraphicsState::create(mpDevice);

    // Set depth function
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::LessEqual).setDepthWriteMask(true);
    mRaster.pState->setDepthStencilState(DepthStencilState::create(dsDesc));

    mpFbo = Fbo::create(mpDevice);
}

Properties DeepOpacityMapPass::getProperties() const
{
    return {};
}

RenderPassReflection DeepOpacityMapPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);
    reflector.addOutput(kVBufferName, kVBufferDesc)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess)
        .format(mVBufferFormat)
        .texture2D(sz.x, sz.y);
    // reflector.addOutput("dst");
    // reflector.addInput("src");
    return reflector;
}

void DeepOpacityMapPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    setLight(mpScene && mpScene->getLightCount() ? mpScene->getLight(0) : nullptr);
    mFrameCount = 0;

    if (pScene)
    {
        // Trigger graph recompilation if we need to change the V-buffer format.
        ResourceFormat format = pScene->getHitInfo().getFormat();
        if (format != mVBufferFormat)
        {
            mVBufferFormat = format;
            requestRecompile();
        }
    }

    if (pScene)
    {
        if (pScene->getCurveVao() && pScene->getCurveVao()->getPrimitiveTopology() != Vao::Topology::LineStrip)
        {
            FALCOR_THROW("DOM pass Raster: Requires LineStrip geometry");
        }
    }
}

void DeepOpacityMapPass::recreatePrograms()
{
    mRaster.pProgram = nullptr;
    mRaster.pVars = nullptr;
}

void DeepOpacityMapPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pOutput = renderData.getTexture(kVBufferName);
    FALCOR_ASSERT(pOutput);
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    // Clear depth and output buffer.
    pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0)); // Clear as UAV for integer clear value

    // If there is no scene, we're done.
    if (mpScene == nullptr)
    {
        return;
    }
    GenerateShadowPass(mpScene->getCamera().get(), (float)(pOutput->getWidth())/(float)(pOutput->getHeight()) );

    // Check for scene changes.
    // if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded))
    // {
    //     recreatePrograms();
    // }

    // Create raster program.
    if (!mRaster.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").gsEntry("gsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mRaster.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mRaster.pState->setProgram(mRaster.pProgram);
    }

    // Create program vars.
    if (!mRaster.pVars)
    {
        mRaster.pVars = ProgramVars::create(mpDevice, mRaster.pProgram.get());
    }

    mpFbo->attachColorTarget(pOutput, 0);
    mRaster.pState->setFbo(mpFbo); // Sets the viewport

    auto var = mRaster.pVars->getRootVar();
    var["DOM"]["lightVP"] = mLightVP;
    var["DOM"]["lightposW"] = mLightPos;

    // Rasterize the scene.
    RasterizerState::CullMode cullMode = kDefaultCullMode;
    mpScene->rasterize(pRenderContext, mRaster.pState.get(), mRaster.pVars.get(), cullMode);
}

void DeepOpacityMapPass::renderUI(Gui::Widgets& widget) {}
