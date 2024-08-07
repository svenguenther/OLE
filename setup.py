from setuptools import setup
from OLE import __author__, __version__

setup(
    name="OLE",
    version=__version__,
    description="Online Learning Emulator",
    url="https://github.com/svenguenther/OLE",
    author=__author__,
    license="...",
    packages=["OLE"],
    package_data={"OLE": ["likelihoods/*", "samplers/*", "utils/*"]},
    zip_safe=False,
)
