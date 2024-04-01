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
#include "CurveShadowPass.h"
#include "Scene/HitInfo.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{

const std::string kProgramComputeFile = "Source/RenderPasses/CurveShadowPass/CurveShadowPass.cs.slang";
const std::string kShadowBufferName = "Shadow map";
const std::string kShadowBufferDesc = "Shadow map visulization";
}; // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, CurveShadowPass>();
}

CurveShadowPass::CurveShadowPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
}

Properties CurveShadowPass::getProperties() const
{
    return {};
}

RenderPassReflection CurveShadowPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    // Define the required resources here
    reflector.addOutput(kShadowBufferName, kShadowBufferDesc)
    .bindFlags(ResourceBindFlags::UnorderedAccess)
    .format(mVBufferFormat)
    .texture2D(sz.x, sz.y);

    return reflector;
}

void CurveShadowPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pOutput = renderData.getTexture(kShadowBufferName); // get output texture pointer
    FALCOR_ASSERT(pOutput);
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    if (mpScene == nullptr)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0)); // unordered access view
        return;
    }

    executeCompute(pRenderContext, renderData);

    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");
    mFrameCount ++;
}

void CurveShadowPass::executeCompute(RenderContext* pRenderContext, const RenderData& renderData){
    if (!mpComputePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules()); //向程序描述中添加场景的着色器模块。
        desc.addShaderLibrary(kProgramComputeFile).csEntry("main"); // 从着色器库文件（kProgramComputeFile）中加载计算着色器，并指定入口函数为"main"。
        desc.addTypeConformances(mpScene->getTypeConformances()); //添加类型一致性声明。

        DefineList defines;
        defines.add(mpScene->getSceneDefines());

        mpComputePass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpComputePass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
        mpSampleGenerator->bindShaderData(var);
    }

    // mpComputePass->getProgram()->addDefines(getShaderDefines(renderData));

    ShaderVar var = mpComputePass->getRootVar();
    bindShaderData(var, renderData);

    mpComputePass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void CurveShadowPass::renderUI(Gui::Widgets& widget) {}

void CurveShadowPass::updateFrameDim(const uint2 frameDim)
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


void CurveShadowPass::bindShaderData(const ShaderVar& var, const RenderData& renderData)
{
    var["gVBufferRT"]["frameDim"] = mFrameDim;
    var["gVBufferRT"]["frameCount"] = mFrameCount;

    // Bind resources.
    var["gVBuffer"] = getOutput(renderData, kShadowBufferName);

    // Bind output channels as UAV buffers.
    // auto bind = [&](const ChannelDesc& channel)
    // {
    //     ref<Texture> pTex = getOutput(renderData, channel.name);
    //     var[channel.texname] = pTex;
    // };
}

