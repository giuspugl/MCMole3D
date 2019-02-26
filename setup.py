import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcmole3d-giuspugl",
    version="0.0.1",
    author="Giuseppe Puglisi",
    author_email="giuse.puglisi@gmail.com",
    description="Monte-Carlo realization of Galactic CO  emission",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giuspugl/MCMole3D",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)
