from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="neural_network_arena",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,  # Includes packages from requirements.txt
    entry_points={
        "console_scripts": [
            "run-arena=main:main",
        ],
    },
)
