add_plugin(DeepOpacityMapPass)

target_sources(DeepOpacityMapPass PRIVATE
    DeepOpacityMapPass.cpp
    DeepOpacityMapPass.h
    DeepOpacityMapPass.3d.slang
)
target_copy_shaders(DeepOpacityMapPass RenderPasses/DeepOpacityMapPass)

target_source_group(DeepOpacityMapPass "RenderPasses")
