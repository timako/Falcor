add_plugin(ExampleRasterPass)

target_sources(ExampleRasterPass PRIVATE
    ExampleRasterPass.cpp
    ExampleRasterPass.h
)

target_source_group(ExampleRasterPass "RenderPasses")
