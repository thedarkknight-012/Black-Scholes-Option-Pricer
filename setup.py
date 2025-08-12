from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "black_scholes",
        ["black_scholes.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="black_scholes",
    version="0.1",
    author="Your Name",
    description="Black–Scholes option pricer with Greeks (C++ / pybind11)",
    ext_modules=ext_modules,
    install_requires=["pybind11"],
    zip_safe=False,
)
