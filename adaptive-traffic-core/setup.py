"""
Setup script for Adaptive Traffic Core.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-traffic-core",
    version="1.0.0",
    description="Core traffic signal control system using Deep Reinforcement Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "gymnasium>=0.29.0",
        "tensorflow>=2.17.0",
        "torch>=2.0.0",
        "adaptive-traffic-common",
    ],
    extras_require={
        "sumo": ["sumolib>=1.18.0", "traci>=1.18.0"],
        "vision": ["adaptive-traffic-vision"],
    },
    entry_points={
        "console_scripts": [
            "adaptive-traffic-train=adaptive_traffic_core.rl.train_dqn:main",
        ],
    },
)

