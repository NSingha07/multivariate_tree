from setuptools import setup, find_packages

setup(
    name="multivariate_tree",
    version="0.1.0",
    description="Multivariate Regression Tree Model Package",
    author="Nitul Singha",
    author_email="nitulsingha07@gmail.com",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "graphviz",
    ],  # List any other dependencies if needed.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
