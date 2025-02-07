from setuptools import setup, find_packages

setup(
    name="EAI-Pers",
    version="0.1.0",  # Initial version
    description="Personalized robotic navigation via querable online maps",  # Short description
    long_description=open("README.md").read(),  # Use README.md as the long description
    long_description_content_type="text/markdown",  # Specify markdown for the long description
    author="Filippo Ziliotto",
    author_email="filippo.ziliotto@phd.unipd.it", 
    url="https://github.com/filippoziliotto/eai-pers.git",  #
    license="MIT",  # Specify your license
    packages=find_packages(),  # Automatically find all packages in the project
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    python_requires=">=3.9",  # Specify the required Python version
    install_requires=[ # Check requirements.txt for the list of dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="computer-vision robotics navigation maps",  # Add relevant keywords
)
