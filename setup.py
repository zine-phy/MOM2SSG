from setuptools import setup, find_packages
import os


setup(
    name="identifySSG",  
    version="1.0.0",  
    description="A toolkit to identify spin space groups (SSGs) for magnetic materials.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",  
    author="Your Name",
    author_email="songziyin@iphy.ac.cn",
    url="https://github.com/zine-phy/MOM2SSG",  
    license="MIT",  
    packages=find_packages(),  
    include_package_data=True,  
    package_data={
        "identifySSG": ["ssg_data/*"],
        "identifySSG.smith_form": ["libintlng.so"],
    },
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "identifySSG=identifySSG.identifySSG:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.7",  
)