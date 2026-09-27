from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vtcv-fine-grained-classification",
    version="0.1.0",
    description="TaxoAttention: taxonomic and semantic streams with attention-guided augmentation for mosquito identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sameerah Talafha, Thomas Jenkins",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "torchvision",
        "pretrainedmodels",
        "numpy",
        "pandas",
        "pillow",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "opencv-python",
        "reportlab",
        "tensorboard",
        "tqdm",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "vtcv-fine-grained-classification = vtcv_fine_grained_classification.__main__:run"
        ]
    },
)
