import torch

from typing import Tuple

from . import model
from . import data
from . import config


def eular_to_vector(eular: torch.Tensor) -> torch.Tensor:
    """
    eular: (N, 2)
    out: (N, 3)
    """
    theta = eular[:, 0]
    phi = eular[:, 1]
    return torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=1)


evaluate_neural_brdf_template = """
Texture2DArray<float> _brdf_latent_texture;

float3 evaluate_neural_brdf(float2 uv, float3 dir_in, float3 dir_out) {
    LargeVector<{{N_FEATURES_IN}}> neural_in;
    
    neural_in.value[0] = dir_in.x;
    neural_in.value[1] = dir_in.y;
    neural_in.value[2] = dir_in.z;

    neural_in.value[3] = dir_out.x;
    neural_in.value[4] = dir_out.y;
    neural_in.value[5] = dir_out.z;
    
    for(uint i = 0; i < {{LATENT_DIMS}}; i++) {
        neural_in.value[i + 6] = _brdf_latent_texture.Sample(_brdf_sampler_default, float3(uv, i));
    }

    let neural_out = exp(forward(neural_in).to_vector()) - 1;
    return max(0, neural_out);
}
"""


class ModelSet:
    def __init__(self, material: data.Material, resolution: Tuple[int, int] = (256, 256), latent_dim: int = 8):
        """
        resolution: height * width
        """ 
        self.material = material
        self.latent_dim = latent_dim
        self.n_features_in = 3
        self.resolution = resolution
        self.encoder = model.MultiLayerPerceptron(n_features_in=self.n_features_in, n_features_out=self.latent_dim, n_features_hidden=64, n_hidden_layers=4).to(config.device())
        self.brdf_decoder = model.MultiLayerPerceptron(n_features_in=self.latent_dim+6, n_features_out=3, n_features_hidden=32, n_hidden_layers=3).to(config.device())

        self.latent_embeddings = None
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.brdf_decoder.parameters()))

    def random_eular(self, count: int):
        eular = torch.zeros(count, 2)
        eular[:, 0] = torch.rand(count) * torch.pi * 2.0 - torch.pi
        eular[:, 1] = torch.rand(count) * torch.pi * 0.5
        return eular

    def sample_end_to_end(self, count: int):
        uv = torch.rand(count, 2).to(config.device())

        dir_in = torch.randn(count, 3).to(config.device())
        dir_out = torch.randn(count, 3).to(config.device())

        dir_in = dir_in / torch.norm(dir_in, dim=1, keepdim=True)
        dir_out = dir_out / torch.norm(dir_out, dim=1, keepdim=True)

        parameters, brdf = self.material.evaluate(uv, dir_in, dir_out)
        
        return parameters, torch.cat([dir_in, dir_out], dim=1), brdf
    
    def sample_finetune(self, count: int):
        rand_uniform = torch.rand(count, 2).to(config.device())

        pos = torch.stack([
            rand_uniform[:, 0] * self.resolution[0],
            rand_uniform[:, 1] * self.resolution[1]
        ], dim=1).type(torch.IntTensor).to(config.device())
        pos[:, 0] = torch.clamp(pos[:, 0], 0, self.resolution[0] - 1)
        pos[:, 1] = torch.clamp(pos[:, 1], 0, self.resolution[1] - 1)

        uv = torch.stack([
            pos[:, 0] / (self.resolution[0] - 1.0),
            pos[:, 1] / (self.resolution[1] - 1.0)
        ], dim=1)

        dir_in = torch.randn(count, 3).to(config.device())
        dir_out = torch.randn(count, 3).to(config.device())

        dir_in = dir_in / torch.norm(dir_in, dim=1, keepdim=True)
        dir_out = dir_out / torch.norm(dir_out, dim=1, keepdim=True)

        _, brdf = self.material.evaluate(uv, dir_in, dir_out)

        latent = self.latent_embeddings((pos[:, 1] * self.resolution[0] + pos[:, 0]))

        return latent, torch.cat([dir_in, dir_out], dim=1), brdf

    def begin_finetuning(self):
        us = torch.linspace(0, 1, self.resolution[0]).repeat(self.resolution[1]).to(config.device())
        vs = torch.linspace(0, 1, self.resolution[1]).repeat_interleave(self.resolution[0]).to(config.device())
        uvs = torch.stack([us, vs], dim=1)
        parameters = self.material.get_parameters(uvs).cpu()
        latent = torch.sigmoid(self.encoder.cpu().forward(parameters.cpu()))
        latent = latent.reshape(self.resolution[0], self.resolution[1], self.latent_dim)

        assert self.latent_embeddings is None
        self.latent_embeddings = torch.nn.Embedding(self.resolution[0] * self.resolution[1], self.latent_dim).to(config.device())
        # self.latent_embeddings.weight.data.copy_(torch.zeros(self.resolution[0] * self.resolution[1], self.latent_dim).to(config.device()))
        self.latent_embeddings.weight.data.copy_(latent.reshape(-1, self.latent_dim).to(config.device()))

        self.optimizer = torch.optim.Adam(list(self.brdf_decoder.parameters())+ list(self.latent_embeddings.parameters()))

    def generate_latent_texture(self):
        return self.latent_embeddings.weight.data.detach().cpu().numpy().reshape(self.resolution[1], self.resolution[0], self.latent_dim)

    def train(self, samples_per_epoch: int, n_epochs: int, lr: float, verbose=False):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            if self.latent_embeddings is None:
                parameters, direction, brdf_value = self.sample_end_to_end(samples_per_epoch)
                latent = torch.sigmoid(self.encoder.forward(parameters))
            else:
                latent, direction, brdf_value = self.sample_finetune(samples_per_epoch)

            log_brdf_pred = self.brdf_decoder.forward(torch.cat([direction, latent], dim=1))
            log_brdf_real = torch.log(brdf_value + 1)

            loss = torch.nn.functional.l1_loss(log_brdf_pred, log_brdf_real)
            loss.backward()

            self.optimizer.step()
            
            if verbose:
                print(f"Epoch {epoch}: loss = {loss.cpu().item()}")

    def generate_code(self):
        out = ""
        out += "import nn;\n"
        out += f"import materials.{self.material.name};\n"
        out += "SamplerState _brdf_sampler_default;\n"
        out += self.brdf_decoder.generate_function("forward")

        out += evaluate_neural_brdf_template \
                .replace("{{N_FEATURES_IN}}", str(self.latent_dim + 6)) \
                .replace("{{LATENT_DIMS}}", str(self.latent_dim))
        
        out += self.material.generate_legacy_code()

        return out
