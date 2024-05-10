add_plugin(HairRenderPass)

target_sources(HairRenderPass PRIVATE
    HairRenderPass.cpp
    HairRenderPass.h
    HairRenderPass.cs.slang
    HairRenderPass.slang
    HairMaterial_re.h
    MathUtils.h
)
target_copy_shaders(HairRenderPass RenderPasses/HairRenderPass)

target_source_group(HairRenderPass "RenderPasses")
