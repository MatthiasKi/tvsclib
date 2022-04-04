from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='tvsclib',
    packages=find_packages(include=['tvsclib']),
    version='0.2.0',
    description='Time varying systems computation library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatthiasKi/tvsclib",    
    author="Daniel Stümke, Stephan Nüßlein, Matthias Kissel",
    author_email="matthias.kissel@tum.de",
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    tests_require=['nose'],
    test_suite='tests',
)
