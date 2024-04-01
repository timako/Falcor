add_plugin(CurveShadowPass)

target_sources(CurveShadowPass PRIVATE
    CurveShadowPass.cpp
    CurveShadowPass.h
    CurveShadowPass.cs.slang
    CurveShadowPass.slang
)

target_source_group(CurveShadowPass "RenderPasses")
