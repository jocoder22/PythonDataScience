# this is the setup file
from setuptools import setup

setup(name = 'mypackage',
      vesion = '0.0.1',
      description = 'This is my first package',
      author = 'Jocoder22',
      author_email = '...@gmail.com',
      packages = ['mypackage'],
      install_requires = ['matplotlib',
                          'numpy',
                          'pandas'])