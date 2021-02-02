import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mc_ocr",
    version="1.0.0",
    author="AI Research Lab - Samsung SDS Vietnam",
    author_email="nd.cuong1@samsung.com",
    description="Project fof MC_OCR competition",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    install_requires=[],
    url="",
    packages=setuptools.find_packages(),
    package_data={'mc_ocr': []},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
