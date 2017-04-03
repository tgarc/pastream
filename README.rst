pastream Portaudio Streams for Python
=======================================

`pastream` builds on top of `portaudio <http://www.portaudio.com/>`_ and the
excellent `sounddevice <http://github.com/spatialaudio/python-sounddevice>`_ python
bindings to provide some more advanced functionality right out of the box.


Features
--------

GIL-less Audio Callbacks
    Having the portaudio callback implemented in C means audio interrupts can be
    serviced quickly and reliably without ever needing to acquire the GIL.

Expanded State Machine
    Adds the ability to differentiate whether a stream has been aborted or
    completed successfully even after the stream has finished.

Input Stream iterators
    Efficiently retrieve live audio capture data through an iterable. Especially
    useful for audio analysis tasks.

Reader/Writer Threads
    pastream simplifies the process of implementing stream reader and writer
    threads to manipulate and/or generate data in the background while leaving
    the main thread free for higher level management tasks.


Dependencies
------------

`sounddevice <http://github.com/spatialaudio/python-sounddevice>`_

`soundfile <https://github.com/bastibe/PySoundFile>`_

(Optional) `numpy <http://www.numpy.org/>`_


Installation
------------

If doing a fresh checkout::

    $ git clone --recursive http://github.com/tgarc/pastream

If you already have a checkout::

    $ git submodule update --init

Then do a pip install::

    $ pip install <path/to/checkout>

Compilation
------------

Note that you will need to have the proper build environment set up in order to compile pastream's C extensions. 

On Mac OSX and unix platforms you'll simply need to have a C compiler installed - no extra development files are required.

For Windows users this `document <https://packaging.python.org/extensions/#setting-up-a-build-environment-on-windows>`_ will guide you through setting up a build environment for your Python version. To sum it up:

For Python 2.7
    Install VS2008 from `here <https://www.microsoft.com/en-gb/download/details.aspx?id=44266>`_

For Python 3.4
    Install Windows SDK for Windows 7 and .NET Framework 4  from `here <https://www.microsoft.com/en-gb/download/details.aspx?id=8279>`_

For Python 3.5
    Install VS2015 from `here <https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx>`_
