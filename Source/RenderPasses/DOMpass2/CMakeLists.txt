add_plugin(DOMpass2)

target_sources(DOMpass2 PRIVATE
    DOMpass2.cpp
    DOMpass2.h
    DOMpass2.3d.slang
)
target_copy_shaders(DOMpass2 RenderPasses/DOMpass2)

target_source_group(DOMpass2 "RenderPasses")
