from PIL import Image

import torch
import numpy as np
import neuralmat
from matplotlib import pyplot as plt

parameters = neuralmat.data.ParametersDescriptor()
parameters.descriptors.append(
    neuralmat.data.ParameterDescriptor(
        name="albedo",
        n_channels=3,
        texture=neuralmat.data.Texture("test.png", srgb=True)
    )
)

resolution = parameters.descriptors[0].image.shape
resolution = (resolution[1], resolution[0])

material = neuralmat.data.SlangShaderMaterial("lambert", parameters)
modelset = neuralmat.train.ModelSet(material, resolution=resolution, latent_dim=3)
modelset.train(samples_per_epoch=100000, n_epochs=3000, lr=0.01,  verbose=True)
modelset.train(samples_per_epoch=100000, n_epochs=3000, lr=0.003, verbose=True)
modelset.train(samples_per_epoch=100000, n_epochs=3000, lr=0.001, verbose=True)
modelset.begin_finetuning()
modelset.train(samples_per_epoch=100000, n_epochs=3000, lr=0.003, verbose=True)
modelset.train(samples_per_epoch=100000, n_epochs=3000, lr=0.001, verbose=True)
modelset.train(samples_per_epoch=100000, n_epochs=3000, lr=0.0003, verbose=True)
modelset.generate_code()

with open(f"render/compiled.slang", "w") as f:
    f.write(modelset.generate_code())

with open(f"render/resource_descs.py", "w") as f:
    source = "from falcor import *\n"
    source += material.generate_texture_descs_code()
    f.write(source)

latent_texture = modelset.generate_latent_texture()
neuralmat.image.write_dds("render/latent.dds", latent_texture)