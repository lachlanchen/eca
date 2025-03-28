from setuptools import setup, find_packages

setup(
    name="eigen_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "torch>=1.7.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
    ],
    author="Lachlan Chen",
    author_email="lach@lazying。art",
    description="Eigencomponent Analysis for classification and clustering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lachlanchen/eigen_analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
