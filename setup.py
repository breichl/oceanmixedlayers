import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="oceanmixedlayers", # Replace with your own username                                            
    version="0.0.1",
    author="Brandon Reichl",
    author_email="brandon.reichl@noaa",
    description="A variety of mld calculations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/breichl/TEMPORARY",
    license = "...",
    project_urls={
        "Bug Tracker": "https://github.com/breichl/TEMPORARY/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: TEMPORARY",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "oceanmixedlayers"},
    #packages=setuptools.find_packages(where="oceanmixedlayers"),
    packages=['oceanmixedlayers'],
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'gsw'
    ],
)
