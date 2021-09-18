from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='synergos',
    version='0.0.1',
    description='Package of ML useful tools',
    py_modules=['synergos'],
    package_dir={'': 'synergos'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas>=1.0.0",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
            "tox>=3.24.3",
            "check-manifest",
            "twine"
        ],
    },
    url='https://github.com/PeterPirog/synergos',
    author='Peter Pirog',
    author_email='peterpirogtf@gmail.com',
)
