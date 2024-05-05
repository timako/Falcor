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
#include "HairRenderPass.h"
#include "HairMaterial.h"
#include "Scene/HitInfo.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
const std::string kProgramComputeFile = "RenderPasses/HairRenderPass/HairRenderPass.cs.slang";
const std::string kShadowBufferName = "Shadow map";
const std::string kShadowBufferDesc = "Shadow map visulization";

const ChannelList kInputChannels = {
    // clang-format off
    { "DOMmap",        "gDOMmap",     "Shadowmap of main light", true /* optional */, ResourceFormat::RGBA32Float },
    // clang-format on
};

const ChannelList kVBufferExtraChannels = {
    // clang-format off
    { "depth",          "gDepth",           "Depth buffer (NDC)",               true /* optional */, ResourceFormat::R32Float    },
    { "mvec",           "gMotionVector",    "Motion vector",                    true /* optional */, ResourceFormat::RG32Float   },
    { "viewW",          "gViewW",           "View direction in world space",    true /* optional */, ResourceFormat::RGBA32Float }, // TODO: Switch to packed 2x16-bit snorm format.
    { "time",           "gTime",            "Per-pixel execution time",         true /* optional */, ResourceFormat::R32Uint     },
    { "mask",           "gMask",            "Mask",                             true /* optional */, ResourceFormat::R32Float    },
    // clang-format on
};

}; // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, HairRenderPass>();
}

void HairRenderPass::setLight(ref<Light> pLight)
{
    if(pLight){
        mpLight = pLight;
    }

}

void HairRenderPass::createShadowMatrix(const PointLight* pLight, const float3 center, float radius, float fboAspectRatio, float4x4& shadowVP){
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

void HairRenderPass::createShadowMatrix(const Light* pLight, const float3& center, float radius, float fboAspectRatio, float4x4& shadowVP)
{
    if(pLight->getType() == LightType::Point){
        PointLight* ppLight = (PointLight*)pLight;
        ppLight->setOpeningAngle((float)6.28319 / (float)8.0); // 设置正常的视场角（45度）
        createShadowMatrix((PointLight*)pLight, center, radius, fboAspectRatio, shadowVP);
    }
}


void HairRenderPass::GenerateShadowPass(const Camera* pCamera, float aspect){
    AABB mpSceneBB = mpScene->getSceneBounds();
    const float3 center((mpSceneBB.minPoint.x + mpSceneBB.maxPoint.x)/2,
                        (mpSceneBB.minPoint.y + mpSceneBB.maxPoint.y)/2,
                        (mpSceneBB.minPoint.z + mpSceneBB.maxPoint.z)/2);

    const float radius = math::length(mpSceneBB.extent()) / 2;

    createShadowMatrix(mpLight.get(), center, radius, aspect, mLightVP);
}

HairRenderPass::HairRenderPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);

    Np_tex = pDevice->createTexture2D(
            AZIMUTHAL_PRECOMPUTE_RESOLUTION,
            AZIMUTHAL_PRECOMPUTE_RESOLUTION * 3,
            ResourceFormat::RGB32Float,
            1,
            1,
            mHairBSDF.Np_table,
            ResourceBindFlags::ShaderResource
        );
}

Properties HairRenderPass::getProperties() const
{
    return {};
}

RenderPassReflection HairRenderPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    // Define the required resources here
    reflector.addOutput(kShadowBufferName, kShadowBufferDesc)
    .bindFlags(ResourceBindFlags::UnorderedAccess)
    .format(mVBufferFormat)
    .texture2D(sz.x, sz.y);

    addRenderPassOutputs(reflector, kVBufferExtraChannels, ResourceBindFlags::UnorderedAccess, sz);

    return reflector;
}

void HairRenderPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pOutput = renderData.getTexture(kShadowBufferName); // get output texture pointer
    FALCOR_ASSERT(pOutput);
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    if (mpScene == nullptr)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0)); // unordered access view
        return;
    }
    GenerateShadowPass(mpScene->getCamera().get(), (float)(pOutput->getWidth())/(float)(pOutput->getHeight()) );
    pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0));
    clearRenderPassChannels(pRenderContext, kVBufferExtraChannels, renderData);

    executeCompute(pRenderContext, renderData);

    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");
    mFrameCount ++;
}

void HairRenderPass::executeCompute(RenderContext* pRenderContext, const RenderData& renderData){
    if (!mpComputePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules()); //向程序描述中添加场景的着色器模块。
        desc.addShaderLibrary(kProgramComputeFile).csEntry("main"); // 从着色器库文件（kProgramComputeFile）中加载计算着色器，并指定入口函数为"main"。
        desc.addTypeConformances(mpScene->getTypeConformances()); //添加类型一致性声明。

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());

        mpComputePass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpComputePass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
        mpSampleGenerator->bindShaderData(var);
    }

    mpComputePass->getProgram()->addDefines(getShaderDefines(renderData));

    ShaderVar var = mpComputePass->getRootVar();
    bindShaderData(var, renderData);

    mpComputePass->execute(pRenderContext, uint3(mFrameDim, 1));
}

DefineList HairRenderPass::getShaderDefines(const RenderData& renderData)
{
    DefineList defines;
    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    defines.add(getValidResourceDefines(kVBufferExtraChannels, renderData));
    return defines;
}

void HairRenderPass::renderUI(Gui::Widgets& widget) {}

void HairRenderPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene){
    mpScene = pScene;

    setLight(mpScene && mpScene->getLightCount() ? mpScene->getLight(0) : nullptr);


}

void HairRenderPass::recreatePrograms()
{
    mpComputePass = nullptr;
}

void HairRenderPass::updateFrameDim(const uint2 frameDim)
{
    FALCOR_ASSERT(frameDim.x > 0 && frameDim.y > 0);
    mFrameDim = frameDim;
    mInvFrameDim = 1.f / float2(frameDim);

}

ref<Texture> getOutput(const RenderData& renderData, const std::string& name)
{
    // This helper fetches the render pass output with the given name and verifies it has the correct size.
    auto pTex = renderData.getTexture(name);
    return pTex;
}


void HairRenderPass::bindShaderData(const ShaderVar& var, const RenderData& renderData)
{
    var["gVBufferRT"]["frameDim"] = mFrameDim;
    var["gVBufferRT"]["frameCount"] = mFrameCount;

    // Camera lightCamera("lightCamera");


    var["gVBufferRT"]["ShadowVP"] = mLightVP;
    var["gVBufferRT"]["lightPos"] = mLightPos;
    // Bind resources.
    var["gVBuffer"] = getOutput(renderData, kShadowBufferName);
    var["gNp_tex"] = Np_tex;

    auto bind = [&](const ChannelDesc& channel)
    {
        ref<Texture> pTex = getOutput(renderData, channel.name);
        var[channel.texname] = pTex;
    };
    for (const auto& channel : kVBufferExtraChannels)
        bind(channel);

    // Bind output channels as UAV buffers.
    // auto bind = [&](const ChannelDesc& channel)
    // {
    //     ref<Texture> pTex = getOutput(renderData, channel.name);
    //     var[channel.texname] = pTex;
    // };
}

