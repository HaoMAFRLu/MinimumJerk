from setuptools import setup, find_packages

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="minimum_jerk",
    version="0.0.2",
    author="Hao Ma",
    author_email="hao.ma@tuebingen.mpg.de",
    description="Path planning using minimum jerk algorithm.",
    long_description=long_description,
    url="https://github.com/HaoMAFRLu/MinimumJerk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)