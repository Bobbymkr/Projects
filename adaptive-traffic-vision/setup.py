"""
Setup script for Adaptive Traffic Vision.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-traffic-vision",
    version="1.0.0",
    description="Computer vision pipeline for traffic analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.10.0",
        "ultralytics>=8.3.0",
        "numpy>=1.26.0",
        "adaptive-traffic-common",
    ],
    package_data={
        "adaptive_traffic_vision": ["models/*.pt"],
    },
    include_package_data=True,
)

