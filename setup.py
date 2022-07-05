import setuptools
 
setuptools.setup(
    name="pyvital",
    version="0.1.21",
    author="VitalLab",
    author_email="vital@snu.ac.kr",
    description="Python Libray for Biosignal Analysis",
    long_description="Python Libray for Biosignal Analysis",
    long_description_content_type="text/markdown",
    url="https://github.com/vitaldb/pyvital",
    install_requires=['numpy','scipy','sanic','PyWavelets'],
    packages=['pyvital', 'pyvital.filters'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)