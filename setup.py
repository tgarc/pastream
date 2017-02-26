import sys
from setuptools import setup, find_packages

__version__ = 'unknown'

# "import" __version__
for line in open('pastream.py'):
    if line.startswith('__version__'):
        exec(line)
        break

with open('README.rst') as readme:
    long_description=readme.read()

requirements = ['soundfile>=0.9.0', 'sounddevice>=0.3.7', 'pa-ringbuffer']
# optionals = ['numpy']

setup(name='pastream',
      version=__version__,
      author='Thomas J. Garcia',
      url='http://github.com/tgarc/pastream',
      author_email='toemossgarcia@gmail.com',
      long_description=long_description,
      entry_points={'console_scripts': ['pastream = pastream:_main']},
      cffi_modules=["pastream_build.py:ffibuilder"],
      install_requires=requirements,
      setup_requires=["cffi>=1.0.0"],
      tests_require=['pytest>=3.0'],
      dependency_links=[
          "git+https://github.com/mgeier/python-pa-ringbuffer.git/@master#egg=pa-ringbuffer"
          ],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
