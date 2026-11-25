from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open("README.md", "r") as fh:
    long_description = fh.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name="minimum-violation-velocity-planner",
    version="2022.1",
    description="Minimum Violation Velocity Planner",
    long_description=long_description,
    url="https://gitlab.lrz.de/0000000001497FF4/minimum-violation-velocity-planner",
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ]
)
