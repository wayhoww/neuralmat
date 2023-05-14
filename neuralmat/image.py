"""
        result = subprocess.run([slangc_path, fileName, '-o', cppOutName, '-target', 'torch-binding' ] + copts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        slangcOutput = result.stderr.decode('utf-8')
        if slangcOutput.strip():
            print(slangcOutput)
        if result.returncode != 0:
            raise RuntimeError(f"compilation failed with error {result.returncode}")
    
"""

import subprocess
from PIL import Image
import sys
import os

def write_dds(path, image):
    os.makedirs('.temp', exist_ok=True)

    for i in range(image.shape[2]):
        Image.fromarray(image[:, :, i]).save(f".temp/_temp_{i}.tiff")

    result = subprocess.run(
        ['neuralmat/bin/texassemble', 'array', '-f', 'R32_FLOAT', '-o', path, '-y'] + [f".temp/_temp_{i}.tiff" for i in range(image.shape[2])]
    )

    if result.returncode != 0:
        raise RuntimeError(f"write failed with error {result.returncode}")
    
    if result.stdout:
        sys.stdout.buffer.write(result.stdout)
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
