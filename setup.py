from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "A low-code recommendation engine generation tool"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="amethyst",
    version=VERSION,
    packages=find_packages('amethyst', exclude=("tests",)),
    install_requires=open("requirements.txt").readlines(),
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/radioactive11/amethyst",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="python machine-learning deep-learning recommendation-engine PyTorch collaborative filtering",
)