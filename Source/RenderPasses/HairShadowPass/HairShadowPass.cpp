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
#include "HairShadowPass.h"


namespace {
    const char kShaderFile[] = "RenderPasses/HairShadowPass/HairShadowPass.3d.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, HairShadowPass>();
}

void camClipSpaceToWorldSpace(const Camera* pCamera, float3 viewFrustum[8], float3& center, float& radius)
{
    float3 clipSpace[8] =
    {
        float3(-1.0f, 1.0f, 0),
        float3(1.0f, 1.0f, 0),
        float3(1.0f, -1.0f, 0),
        float3(-1.0f, -1.0f, 0),
        float3(-1.0f, 1.0f, 1.0f),
        float3(1.0f, 1.0f, 1.0f),
        float3(1.0f, -1.0f, 1.0f),
        float3(-1.0f, -1.0f, 1.0f),
    };

    float4x4 invViewProj = pCamera->getInvViewProjMatrix();
    center = float3(0, 0, 0);

    for (uint32_t i = 0; i < 8; i++)
    {
        float4 crd = math::mul(invViewProj , float4(clipSpace[i], 1));
        viewFrustum[i] = float3(crd.x, crd.y, crd.z) / crd.w;
        center += viewFrustum[i];
    }

    center *= (1.0f / 8.0f);

    // Calculate bounding sphere radius
    radius = 0;
    for (uint32_t i = 0; i < 8; i++)
    {
        float d = math::length(center - viewFrustum[i]);
        radius = std::max(d, radius);
    }
}

static void createShadowMatrix(const PointLight* pLight, const float3& center, float radius, float fboAspectRatio, float4x4& shadowVP)
{
    const float3 lightPos = pLight->getWorldPosition();
    const float3 lookat = pLight->getWorldDirection() + lightPos;
    float3 up(0, 1, 0);
    if (abs(dot(up, pLight->getWorldDirection())) >= 0.95f)
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

static void createShadowMatrix(const Light* pLight, const float3& center, float radius, float fboAspectRatio, float4x4& shadowVP)
{
    switch (pLight->getType())
    {
    case LightType::Point:
        return createShadowMatrix((PointLight*)pLight, center, radius, fboAspectRatio, shadowVP);
    default:
        FALCOR_UNREACHABLE();
    }
}

void HairShadowPass::setLight(ref<Light> pLight)
{
    mpLight = pLight;
}

void HairShadowPass::GenerateShadowPass(const Camera* pCamera){
    struct
    {
        float3 coords[8];
        float3 center;
        float radius;
    } camFrustum;

    camClipSpaceToWorldSpace(pCamera, camFrustum.coords, camFrustum.center, camFrustum.radius);

    createShadowMatrix(mpLight.get(), camFrustum.center, camFrustum.radius, mShadowPass.fboAspectRatio, mHSPData.globalMat);

    float nearPlane = pCamera->getNearPlane();
    float farPlane = pCamera->getFarPlane();
    float depthRange = farPlane - nearPlane;


}

HairShadowPass::HairShadowPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    RasterizerState::Desc wireframeDesc;
    wireframeDesc.setFillMode(RasterizerState::FillMode::Wireframe);
    wireframeDesc.setCullMode(RasterizerState::CullMode::Back);
    mpRasterState = RasterizerState::create(wireframeDesc);

    mpGraphicsState = GraphicsState::create(mpDevice);

    mpFbo = Fbo::create(mpDevice);
}

Properties HairShadowPass::getProperties() const
{
    return {};
}

RenderPassReflection HairShadowPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addOutput("output", "Wireframe view texture");
    return reflector;
}

void HairShadowPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pTex = renderData.getTexture("output");
    mpFbo->attachColorTarget(pTex, uint32_t(0));
    const float4 clearColor(0, 0, 0, 1);
    pRenderContext->clearFbo(mpFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mpGraphicsState->setFbo(mpFbo);


    if (mpScene) {
        mpVars->getRootVar()["PerFrameCB"]["gColor"] = float4(0, 1, 0, 1);
        mpScene->rasterize(pRenderContext, mpGraphicsState.get(), mpVars.get(), mpRasterState, mpRasterState);
    }
}

void HairShadowPass::renderUI(Gui::Widgets& widget) {}

void HairShadowPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {
    mpScene = pScene;

    setLight(mpScene && mpScene->getLightCount() ? mpScene->getLight(0) : nullptr);

    if (mpScene && mpProgram == nullptr) {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mpProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mpGraphicsState->setProgram(mpProgram);

        mpVars = ProgramVars::create(mpDevice, mpProgram.get());
    }

}
