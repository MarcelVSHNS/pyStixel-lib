from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyStixel-lib',
    version='0.2.1',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.9',
    url='https://github.com/MarcelVSHNS/pyStixel-lib',
    license='Apache License 2.0',
    author='Marcel',
    author_email='marcel.vosshans@hs-esslingen.de',
    description='The Python Stixel Library',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
