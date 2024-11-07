from setuptools import setup, find_packages

setup(
    name="raddest",
    version="0.5.0",
    author="Tao Jing",
    author_email="jingt20@mails.tsinghua.edu.cn",
    description=
    "reGression for Astronomical data with Realistic Distributions, Errors and Non-linearity",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy', 'joblib', 'pzflow', 'optax', 'interpax', 'scipy',
        'statsmodels'
    ])
