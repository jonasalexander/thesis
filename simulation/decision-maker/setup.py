from setuptools import setup

setup(
    name="Decision-Maker",
    version="0.1.0",
    author="Jonas Nelle",
    author_email="jonas.a.nelle@gmail.com",
    packages=["dmaker"],
    license="LICENSE.txt",
    description="Object oriented approach to simulating decision making under uncertainty",
    long_description=open("README.txt").read(),
    install_requires=[
        "pandas == 1.2.*",
        "seaborn == 0.11.*",
        "matplotlib == 3.4.*",
        "numpy == 1.19.*",
    ],
)
