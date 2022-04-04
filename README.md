# Time Varying Systems and Computation (TVSC) Library

This is a python library for computations with time varying systems (TVSC). The algorithms are based on the material of the lecture Time-Varying Systems and Computations held by the chair of data processing at the Technical University of Munich, as well as Prof. Dewildes book with the same title. 

The package inter alia contains
- Implementations for causal, anti causal and mixed time varying systems
- Algorithms for approximating transfer operators with sequentially semiseparable matrices (using Hankel Norm Reduction)
- Algorithms for performing operations with linear time varying systems, such as addition, multiplication and inversion

## Installation

After cloning or downloading the repo from GitHub, the library can be installed by running 

``python3 -m pip install -e .``

from the folder, in which the ``setup.py`` file lies. The ``-e`` attribute stands for editable, which means that changes pulled later from the repo are directly updated to the installation.

## Running Tests

You can test your installation by running unit tests. For that, run

``python3 setup.py test``

All tests should be passed if your installation is OK. 

## Credits

Thanks to Daniel Stümke for providing the base of this package. Moreover, we would like to thank Stephan Nüßlein, who also advanced the development of this package.
