from setuptools import setup, find_packages

__version__ = 'unknown'

# "import" __version__
for line in open('pastream.py'):
    if line.startswith('__version__'):
        exec(line)
        break

setup(name='pastream',
      version=__version__,
      author='Thomas J. Garcia',
      url='http://github.com/tgarc/pastream',
      author_email='toemossgarcia@gmail.com',
      long_description=open('README.rst').read(),
      entry_points={'console_scripts': ['pastream = pastream:_main']},
      cffi_modules=["pastream_build.py:ffibuilder"],
      install_requires=[
          'soundfile>=0.9.0',
          'sounddevice>=0.3.7',
          'pa-ringbuffer'
          ],
      setup_requires=["cffi>=1.0.0"],
      tests_require=['pytest>=3.0', 'numpy'],
      extras_require={'numpy': 'numpy'},
      dependency_links=[
          "git+https://github.com/mgeier/python-pa-ringbuffer.git@master#egg=pa-ringbuffer-0"
          ],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
