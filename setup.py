from setuptools import setup, find_packages


# Read the requirements from the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()

authors = {
    "Asaad Almutareb": "asaad.almutareb@artiquare.com",
    "Isayah Culbertson": "isayah@artiquare.com",
}


setup(
    name="InnovationPathFinder",
    version="0.1.0",
    description="A short description of your project",
    long_description="A longer description of your project",
    authors=list(authors.keys()),
    emails=list(authors.values()),
    url="https://github.com/almutareb/InnovationPathfinderAI",
    packages=find_packages(),  # automatically find all packages in the project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # specify the minimum Python version required
    install_requires=read_requirements(),  # read dependencies from requirements.txt
)
