"""
Setup script for Adaptive Traffic Signal Control System.

This package provides an intelligent traffic signal control system using
Deep Reinforcement Learning with Multi-Agent support and computer vision.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-traffic",
    version="1.0.0",
    author="Adaptive Traffic Team",
    author_email="contact@adaptive-traffic.com",
    description="Intelligent Traffic Signal Control using Deep Reinforcement Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adaptive-traffic/adaptive-traffic",
    project_urls={
        "Bug Reports": "https://github.com/adaptive-traffic/adaptive-traffic/issues",
        "Source": "https://github.com/adaptive-traffic/adaptive-traffic",
        "Documentation": "https://adaptive-traffic.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=1.0.0",
        ],
        "full": [
            "wandb>=0.15.0",
            "tensorboardX>=2.6.0",
            "numba>=0.57.0",
            "joblib>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-traffic-train=src.rl.train_dqn:main",
            "adaptive-traffic-inference=src.rl.inference:main",
            "adaptive-traffic-visualize=src.rl.visualize_sim:main",
            "adaptive-traffic-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.xml", "*.sumocfg", "*.pt"],
    },
    keywords=[
        "traffic",
        "signal",
        "control",
        "reinforcement-learning",
        "deep-learning",
        "computer-vision",
        "sumo",
        "multi-agent",
        "optimization",
    ],
    zip_safe=False,
)
