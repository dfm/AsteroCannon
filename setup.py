import os
import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

libraries = []
if os.name == "posix":
    libraries.append("m")
include_dirs = [
    numpy.get_include(),
]

ext = cythonize([
    Extension("acannon._filter",
              sources=["acannon/_filter.pyx"],
              libraries=libraries, include_dirs=include_dirs)
])

setup(
    name="acannon",
    version="0.0.1",
    author="Daniel Foreman-Mackey",
    url="https://github.com/dfm",
    license="MIT",
    packages=["acannon", ],
    ext_modules=ext,
    # description="Blazingly fast Gaussian Processes for regression.",
    # long_description=open("README.rst").read(),
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
