add_plugin(HairShadowPass)

target_sources(HairShadowPass PRIVATE
    HairShadowPass.cpp
    HairShadowPass.h
)

target_source_group(HairShadowPass "RenderPasses")
