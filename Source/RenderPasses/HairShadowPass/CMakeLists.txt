add_plugin(HairShadowPass)

target_sources(HairShadowPass PRIVATE
    HairShadowPass.cpp
    HairShadowPass.h
    HairShadowPass.3d.slang
    HSPData.slang
)
target_copy_shaders(HairShadowPass RenderPasses/HairShadowPass)

target_source_group(HairShadowPass "RenderPasses")
