from setuptools import setup, find_packages

setup(
    name="markov-rl-api-cache",
    version="0.1.0",
    description="Markov Chain-based Reinforcement Learning framework for adaptive API caching in microservices",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[],
)

