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
      cffi_modules=["build_pastream.py:ffibuilder"],
      install_requires=[
          'cffi>=1.0.0',
          'soundfile>=0.9.0',
          'sounddevice>=0.3.7',
          ],
      setup_requires=['CFFI>=1.4.0'],
      tests_require=['pytest>=3.0', 'numpy'],
      extras_require={'numpy': 'numpy'},
      py_modules=['pastream', 'pa_ringbuffer'],
      include_package_data=True,
      zip_safe=False)
