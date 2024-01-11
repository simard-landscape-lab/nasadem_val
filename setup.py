from distutils.core import setup
from os import path

file_dir = path.abspath(path.dirname(__file__))

# To be filled in later
REQUIRES = []

setup(
    name='nasadem_val',
    version='0.1dev',

    description='Tools for Nasadem Validation',
    url='jpl github',

    author='Charlie Marshak',
    author_email='charlie.z.marshak@jpl.nasa.gov',


    keywords='Nasadem GEDI ICESat-2 Peckel Hansen GLIMS',

    packages=['nasadem_val'],  # setuptools.find_packages(exclude=['doc']),

    # Required Packages
    install_requires=REQUIRES,
)
