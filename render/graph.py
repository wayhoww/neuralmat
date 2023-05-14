from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph("DefaultRenderGraph")

    loadRenderPassLibrary("DebugPasses.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ScriptableFullScreenPass.dll")

    from render import resource_descs

    GBufferRaster = createPass(
        "GBufferRaster",
        {
            "outputSize": IOSize.Default,
            "samplePattern": SamplePattern.Center,
            "sampleCount": 16,
            "useAlphaTest": True,
            "adjustShadingNormals": True,
            "forceCullMode": False,
            "cull": CullMode.CullBack,
        },
    )
    g.addPass(GBufferRaster, "GBufferRaster")
    ScriptableFullScreenPass = createPass(
        "ScriptableFullScreenPass",
        {
            "kShaderPath": "render.ps.slang",
            "kResources": [
                ResourceDesc(
                    identifier="Position",
                    type=ResourceDesc.Type.Texture2D,
                    size=uint3(1, 1, 1),
                    autoSized=True,
                    targetSlot=0,
                    view=ResourceDesc.View.SRV,
                    format=ResourceDesc.Format.RGBA32F,
                    clear=False,
                    optional=True,
                ),
                ResourceDesc(
                    identifier="Normal",
                    type=ResourceDesc.Type.Texture2D,
                    size=uint3(1, 1, 1),
                    autoSized=True,
                    targetSlot=0,
                    view=ResourceDesc.View.SRV,
                    format=ResourceDesc.Format.RGBA32F,
                    clear=False,
                    optional=True,
                ),
                ResourceDesc(
                    identifier="Tangent",
                    type=ResourceDesc.Type.Texture2D,
                    size=uint3(1, 1, 1),
                    autoSized=True,
                    targetSlot=0,
                    view=ResourceDesc.View.SRV,
                    format=ResourceDesc.Format.RGBA32F,
                    clear=False,
                    optional=True,
                ),
                ResourceDesc(
                    identifier="TexCoord",
                    type=ResourceDesc.Type.Texture2D,
                    size=uint3(1, 1, 1),
                    autoSized=True,
                    targetSlot=0,
                    view=ResourceDesc.View.SRV,
                    format=ResourceDesc.Format.RG32F,
                    clear=False,
                    optional=True,
                ),
                ResourceDesc(
                    identifier="Target_NN",
                    type=ResourceDesc.Type.Texture2D,
                    size=uint3(1, 1, 1),
                    autoSized=True,
                    targetSlot=0,
                    view=ResourceDesc.View.RTV_Out,
                    format=ResourceDesc.Format.Auto,
                    clear=True,
                    optional=True,
                ),
                ResourceDesc(
                    identifier="Target_Reference",
                    type=ResourceDesc.Type.Texture2D,
                    size=uint3(1, 1, 1),
                    autoSized=True,
                    targetSlot=1,
                    view=ResourceDesc.View.RTV_Out,
                    format=ResourceDesc.Format.Auto,
                    clear=True,
                    optional=True,
                ),
            ],
            "kExternalResources": [
                ExternalTextureDesc(
                    identifier="_brdf_latent_texture", path="latent.dds"
                )
            ] + resource_descs.legacy_texture_descs,
            "kThreads": uint3(1, 1, 1),
            "kCompute": False,
            "kAutoThreads": False,
        },
    )
    g.addPass(ScriptableFullScreenPass, "ScriptableFullScreenPass")
    SplitScreenPass = createPass(
        "SplitScreenPass",
        {
            "splitLocation": 0.5,
            "showTextLabels": True,
            "leftLabel": "Left side",
            "rightLabel": "Right side",
        },
    )
    g.addPass(SplitScreenPass, "SplitScreenPass")
    g.addEdge("GBufferRaster.posW", "ScriptableFullScreenPass.Position")
    g.addEdge("GBufferRaster.normW", "ScriptableFullScreenPass.Normal")
    g.addEdge("GBufferRaster.tangentW", "ScriptableFullScreenPass.Tangent")
    g.addEdge("GBufferRaster.texC", "ScriptableFullScreenPass.TexCoord")
    g.addEdge("ScriptableFullScreenPass.Target_NN", "SplitScreenPass.leftInput")
    g.addEdge("ScriptableFullScreenPass.Target_Reference", "SplitScreenPass.rightInput")
    g.markOutput("SplitScreenPass.output")
    return g


DefaultRenderGraph = render_graph_DefaultRenderGraph()
try:
    m.addGraph(DefaultRenderGraph)
except NameError:
    None
