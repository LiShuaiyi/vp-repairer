import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="commonroad_repair",
    version="0.0.1",
    author="Cyber-Physical Systems Group, Technical University of Munich",
    author_email="commonroad@lists.lrz.de",
    description="It's pip... with git.",
    long_description=long_description,
    url="https://gitlab.lrz.de/yuanfei/commonroad_repair",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD License",
        "Operating System :: OS Independent",
    ],
)
