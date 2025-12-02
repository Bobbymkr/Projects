"""
Setup script for Adaptive Traffic Research.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-traffic-research",
    version="1.0.0",
    description="Research platform for novel traffic control algorithms",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "adaptive-traffic-core",
        "adaptive-traffic-common",
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
    ],
    extras_require={
        "federated": [
            "flwr>=1.5.0",
        ],
    },
)

