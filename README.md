# pyStixel-lib
The Python Stixel Library is a versatile Python toolkit designed for advanced computer vision tasks.
It provides definitions, im-/exports and functions for stixels—vertical strips that represent depth
and scene structure in images. It also introduces the `.stx1` format as a binary export for Stixel Worlds, 
which contains the Stixel World, camera calibration parameters, and the corresponding image.

The basic functionality is to provide some objects and relative functions. The base for Stixel
operations is the Stixel World. The `util` package provide simple draw functions like:
![Sample Stixel World on 2d image plane](https://raw.githubusercontent.com/MarcelVSHNS/pyStixel-lib/main/docs/imgs/Stixel_on_image.png)

Or a RGN supported 3D transformation with provided Camera matrix:
![Sample Stixel World in 3D](https://raw.githubusercontent.com/MarcelVSHNS/pyStixel-lib/main/docs/imgs/pseudo_3d_Stixel.png)

## Usage
A basic usage of the library can be found on 
[Google Colab](https://colab.research.google.com/drive/1ATMEjQMO3QBj6P5EkRAx1-J-6gAVpRmB?usp=sharing).
