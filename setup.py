from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

VERSION = '0.1.3.0'
DESCRIPTION = 'An accurate calculation of technical analysis indicators with values aligning with those in TradingView.'

setup(
    name="tradingview_indicators",
    version=VERSION,
    author="Archie Marques",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords=['python', 'tradingview', 'technical analysis', 'indicators']
)
