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

    LargeVector<3> neural_out = tanh(forward(neural_in));
    return max(0, neural_out.to_vector());
}
"""


class ModelSet:
    def __init__(self, material: data.Material, resolution: Tuple[int, int] = (256, 256), latent_dim: int = 8):
        self.material = material
        self.latent_dim = latent_dim
        self.n_features_in = 3
        self.resolution = resolution
        self.encoder = model.MultiLayerPerceptron(n_features_in=self.n_features_in, n_features_out=self.latent_dim, n_features_hidden=64, n_hidden_layers=4).to(config.device())
        self.brdf_decoder = model.MultiLayerPerceptron(n_features_in=self.latent_dim+6, n_features_out=3, n_features_hidden=32, n_hidden_layers=3).to(config.device())

        # TODO: finetunning
        self.latent_texture = torch.nn.Embedding(self.resolution[0] * self.resolution[1], self.latent_dim).to(config.device())
        
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
    
    def encode_embeddings(self):
        us = torch.linspace(0, 1, self.resolution[0]).repeat(self.resolution[1]).to(config.device())
        vs = torch.linspace(0, 1, self.resolution[1]).repeat_interleave(self.resolution[0]).to(config.device())
        uvs = torch.stack([us, vs], dim=1)
        parameters = self.material.get_parameters(uvs).cpu()
        latent = torch.tanh(self.encoder.cpu().forward(parameters.cpu()))
        latent = latent.reshape(self.resolution[0], self.resolution[1], self.latent_dim)
        latent_np = latent.cpu().detach().numpy()
        return latent_np

    def train(self, samples_per_epoch: int, n_epochs: int, lr: float, verbose=False):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            parameters, direction, brdf_value = self.sample_end_to_end(samples_per_epoch)
            latent = torch.tanh(self.encoder.forward(parameters))
            brdf_pred = torch.tanh(self.brdf_decoder.forward(torch.cat([direction, latent], dim=1)))

            loss = torch.nn.functional.mse_loss(brdf_pred, brdf_value)
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
