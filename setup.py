import os
import re

from setuptools import setup


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(*names, **kwargs):
    with open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)


CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

setup(
    name="pgeocode",
    description="Postal code geocoding",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version=find_version("pgeocode.py"),
    author="Roman Yurchak",
    author_email="roman.yurchak@symerio.com",
    py_modules=["pgeocode"],
    python_requires=">=3.10",
    install_requires=["requests", "numpy", "pandas"],
    extras_require={
        "fuzzy": ["thefuzz"],
    },
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    license="BSD",
)
