# 给定 UV、角度，输出 BRDF
# 向量化运算~~

import torch
import os
import numpy as np
from PIL import Image
import pkg_resources
from . import slangpy
from typing import List
from . import config


package_dir = pkg_resources.resource_filename(__name__, '')


evaluate_legacy_brdf_template = """
{{TEXTURES}}

float3 evaluate_legacy_brdf(float2 uv, float3 dir_in, float3 dir_out) {
    Parameters params;
    
    {{ASSIGN_PARAMETERS}}
    
    return max(0, evaluate_brdf(dir_in, dir_out, params));
}
"""


class Material:
    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, uv: torch.Tensor, di: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
        """
        uv: (N, 2)
        wi: (N, 3), normalized
        wo: (N, 3), normalized
        out: (N, 3)
        """
        raise NotImplementedError()
    
    def get_parameters(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: (N, 2)
        out: (N, D)
        """
        raise NotImplementedError()
    
    def generate_legacy_code(self):
        pass

    

class Texture:
    def __init__(self, path: str, srgb: bool):
        self.path = path
        self.srgb = srgb
        
        image_np = np.array(Image.open(self.path)) / 255.0
        self.image = torch.tensor(image_np)
        if self.srgb:
            self.image = self.image.pow(2.4)

        self.image = self.image.to(config.device())

    @property
    def n_channels(self):
        return self.image.shape[2]


class ParameterDescriptor:
    def __init__(self, name: str, n_channels: int, texture: Texture):
        self.name = name
        self.n_channels = n_channels
        self.texture = texture

        assert self.texture.n_channels >= self.n_channels

    @property
    def image(self):
        return self.texture.image

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: (N, 2)
        out: (N, n_channels)
        """
        uv = uv.clamp(0, 1)
        
        pos = uv * (torch.tensor([self.image.shape[1], self.image.shape[0]]).reshape(1, 2).to(config.device()) - 1)
        
        pos_00 = pos.type(torch.LongTensor)
        pos_00[:, 0] = torch.clamp(pos_00[:, 0], 0, self.image.shape[1] - 2).to(config.device())
        pos_00[:, 1] = torch.clamp(pos_00[:, 1], 0, self.image.shape[0] - 2).to(config.device())
        pos_00 = pos_00.to(config.device())

        pos_01 = pos_00 + torch.tensor([[0, 1]]).to(config.device())
        pos_10 = pos_00 + torch.tensor([[1, 0]]).to(config.device())
        pos_11 = pos_00 + torch.tensor([[1, 1]]).to(config.device())

        o_00 = self.image[pos_00[:, 1], pos_00[:, 0], :self.n_channels]
        o_01 = self.image[pos_01[:, 1], pos_01[:, 0], :self.n_channels]
        o_10 = self.image[pos_10[:, 1], pos_10[:, 0], :self.n_channels]
        o_11 = self.image[pos_11[:, 1], pos_11[:, 0], :self.n_channels]
        
        residual = pos - pos_00.type(torch.FloatTensor).to(config.device())

        w_00 = (1 - residual[:, 0]) * (1 - residual[:, 1])
        w_01 = (1 - residual[:, 0]) * residual[:, 1]
        w_10 = residual[:, 0] * (1 - residual[:, 1])
        w_11 = residual[:, 0] * residual[:, 1]

        w_00 = w_00.reshape(-1, 1)
        w_01 = w_01.reshape(-1, 1)
        w_10 = w_10.reshape(-1, 1)
        w_11 = w_11.reshape(-1, 1)

        return (w_00 * o_00 + w_01 * o_01 + w_10 * o_10 + w_11 * o_11).type(torch.FloatTensor).to(config.device())


class ParametersDescriptor:
    def __init__(self):
        self.descriptors: List[ParameterDescriptor] = []

    @property
    def n_channels(self):
        return sum([d.n_channels for d in self.descriptors])

    def get_parameters(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: (N, 2)
        out: (N, D)
        """
        return torch.cat([d.sample(uv) for d in self.descriptors], dim=1)
    
    def generate_code(self):
        code = "Parameters params;\n"
        channel_id = 6
        for d in self.descriptors:
            for channel in 'xyzw'[:d.n_channels]:
                code += f"    params.{d.name}.{channel} = input[globalIdx.x, {channel_id}];\n"
                channel_id += 1
        return code


class SlangShaderMaterial(Material):
    def __init__(self, material_name, parameters: ParametersDescriptor):
        super().__init__(material_name)

        self.material_name = material_name
        self.parameters = parameters

        with open(os.path.join(package_dir, "cuda_wrapper.slang.template"), "r") as f:
            wrapper_source = f.read()

        wrapper_source = wrapper_source.replace("{{CONSTRUCT_STRUCT}}", self.parameters.generate_code())

        source = f"import materials.{self.material_name};\n"
        source += wrapper_source

        temp_path = os.path.join(package_dir, ".slangpy_cache/cuda_wrapper.slang")
        with open(temp_path, "w") as f:
            f.write(source)

        self.module = slangpy.loadModule(temp_path, copts=["-Irender", "-IC:/Users/wweih/Documents/Falcor/build/windows-vs2022-d3d12/bin/Release/Shaders"])

    def evaluate(self, uv: torch.Tensor, di: torch.Tensor, do: torch.Tensor, verbose=False) -> torch.Tensor:
        parameters = self.get_parameters(uv)
        stacked = torch.cat([di, do, parameters], dim=1)
        return parameters, self.module.evaluate(stacked).to(config.device())

    def get_parameters(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: (N, 2)
        out: (N, D)
        """
        return self.parameters.get_parameters(uv)

    def generate_legacy_code(self):
        
        texture_desc_list = []
        
        for desc in self.parameters.descriptors:
            tex = desc.texture
            tex_name = f"_legacy_tex_{desc.name}"
            texture_desc_list.append(f"Texture2D<float{tex.n_channels}> {tex_name};")

        out = evaluate_legacy_brdf_template
        out = out.replace("{{TEXTURES}}", "\n".join(texture_desc_list))

        assign_list = []
        for desc in self.parameters.descriptors:
            tex_name = f"_legacy_tex_{desc.name}"
            assign_list.append(f"params.{desc.name} = {tex_name}.Sample(_brdf_sampler_default, uv).{'xyzw'[:desc.n_channels]};")

        out = out.replace("{{ASSIGN_PARAMETERS}}", "\n".join([val if i == 0 else '    ' + val for i, val in enumerate(assign_list)]))
        return out
    
    def generate_texture_descs_code(self):
        descs = []
        for desc in self.parameters.descriptors:
            tex = desc.texture
            tex_path = os.path.join(os.getcwd(), tex.path).replace('\\', '/')
            tex_name = f"_legacy_tex_{desc.name}"
            item = f"""ExternalTextureDesc(identifier="{tex_name}", path="{tex_path}", sRGB={str(tex.srgb)})"""
            descs.append(item)
        return "legacy_texture_descs = [" + ", ".join(descs) + "]"
