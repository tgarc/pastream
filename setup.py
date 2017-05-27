import os
from setuptools import setup, find_packages

__version__ = 'unknown'

# "import" __version__
dirname = os.path.dirname(__file__)
for line in open(os.path.join(dirname, 'pastream.py')):
    if line.startswith('__version__'):
        exec(line)
        break

setup(name='pastream',
      version=__version__,
      py_modules=['pastream'],
      author='Thomas J. Garcia',
      url='http://github.com/tgarc/pastream',
      platforms='any',
      license='MIT',
      author_email='toemossgarcia@gmail.com',
      long_description=open(os.path.join(dirname, 'README.rst')).read(),
      entry_points={'console_scripts': ['pastream = pastream:_main']},
      cffi_modules=["build_pastream.py:ffibuilder"],
      install_requires=[
          'cffi>=1.0.0',
          'soundfile>=0.9.0',
          'sounddevice>=0.3.7',
      ],
      setup_requires=['CFFI>=1.4.0'],
      tests_require=['pytest>=3.0', 'numpy'],
      extras_require={'examples': ['numpy', 'matplotlib']},
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Multimedia :: Sound/Audio'
      ],
      include_package_data=True,
      zip_safe=False)
