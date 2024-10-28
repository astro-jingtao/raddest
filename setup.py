from setuptools import setup, find_packages

setup(
    name="rarden",
    version="0.1.0",
    author="Tao Jing",
    author_email="jingt20@mails.tsinghua.edu.cn",
    description=
    "Regression for Astronomical data with Realistic Distributions, Errors and Non-linearity",
    packages=find_packages(),
    install_requires=[
        'numpy', 'joblib', 'pzflow', 'optax', 'scipy', 'statsmodels'
    ])
