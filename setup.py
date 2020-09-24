from setuptools import setup

setup(
    name='cst_transform',
    version='0.1',
    packages=["cst_transform"],
    python_requires=">=3.6",
    install_requires=[
        # "apex @ git+https://github.com/NVIDIA/apex.git#egg=apex"  # apex does not encode dependency on torch
        "numpy",
        "torch",
        "tqdm",
        "transformers",
        "wandb"
    ]
)
