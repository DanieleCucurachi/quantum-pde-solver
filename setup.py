from setuptools import setup, find_packages

setup(
    name="quantum-pde-solver",
    version="0.1.0",
    description="Quantum and classical solvers for PDEs using variational quantum circuits and FiPy.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/quantum-pde-solver",
    packages=find_packages(where=".", exclude=["examples*", "notebooks*", "tests*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "qiskit",
        "qiskit-aer",
        "pandas",
        "fipy",
        "typing_extensions"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
)
