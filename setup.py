from setuptools import find_packages, setup

setup(
    name="raddest",
    version="0.10.0",
    author="Tao Jing",
    author_email="jingt20@mails.tsinghua.edu.cn",
    description=
    "reGression for Astronomical data with Realistic Distributions, Errors and Non-linearity",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy', 'joblib', 'pzflow', 'optax', 'interpax', 'scipy',
        'statsmodels', 'emcee'
    ],
    include_package_data=True,
    package_data={
        'raddest.skills': [
            'SKILL-DEPLOY.md',
            'raddest-regression/meta.yaml',
            'raddest-regression/skill/SKILL.md',
            'raddest-regression/skill/CHANGELOG.md',
            'raddest-regression/skill/references/*.md',
        ]
    })
