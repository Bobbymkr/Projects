"""
Setup script for Adaptive Traffic Common.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-traffic-common",
    version="1.0.0",
    description="Shared utilities and common code for Adaptive Traffic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "pydantic>=2.7.0",
    ],
)

