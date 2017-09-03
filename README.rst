.. image:: https://badge.fury.io/py/pastream.svg
    :target: https://badge.fury.io/py/pastream

.. image:: https://travis-ci.org/tgarc/pastream.svg?branch=master
    :target: https://travis-ci.org/tgarc/pastream

.. image:: https://ci.appveyor.com/api/projects/status/wk52r5jy9ri7dsi9/branch/master?svg=true
    :target: https://ci.appveyor.com/project/tgarc/pastream/branch/master


GIL-less Portaudio Streams for Python
=================================================
`pastream` builds on top of `portaudio <http://www.portaudio.com/>`__ and the
excellent `sounddevice <http://github.com/spatialaudio/python-sounddevice>`__
python bindings to provide some more advanced functionality right out of the
box. Note that in addition to the pastream *library*, pastream includes a
`command line application`_ for playing
and recording audio files.


Features
--------
GIL-less Audio Callbacks
    Having the portaudio callback implemented in C means audio interrupts can
    be serviced quickly and reliably without ever needing to acquire the Python
    Global Interpreter Lock (GIL). This is crucial when working with libraries
    like `Pillow <https://python-pillow.org/>`__ which may greedily grab and
    hold the GIL subsequently causing audio overruns/underruns.

Input Stream iterators
    Efficiently retrieve live audio capture data through an iterable. As simple as:

    .. code-block:: python

       import pastream as ps
       for chunk in ps.chunks():
           process(chunk)

    See ``pastream.chunks`` and ``pastream.InputStream.chunks`` method.

Reader/Writer Threads
    pastream simplifies the process of implementing stream reader and writer
    threads to manipulate and/or generate data in the background while leaving
    the main thread free for higher level management tasks.


Dependencies
------------
`cffi <https://cffi.readthedocs.io/en/latest/>`__

`sounddevice <http://github.com/spatialaudio/python-sounddevice>`__ (depends on `PortAudio <http://www.portaudio.com>`__)

`soundfile <https://github.com/bastibe/PySoundFile>`__ (depends on `libsndfile <http://www.mega-nerd.com/libsndfile/>`__)

(Optional) `numpy <http://www.numpy.org/>`__


Installation
------------
For linux platforms a recent version of the ``PortAudio`` and ``libsndfile`` C
libraries are required. (For Windows and OSX, the sounddevice and soundfile
packages include prebuilt versions for you). You can either install the latest
available from your package manager (e.g. ``apt-get install libportaudio2
libsndfile`` for debian/raspbian) or install the latest stable build from the
package website (Recommended); see links in `Dependencies`_.

pastream is now available on PyPI. Installation is as easy as::

    $ pip install pastream


Building From Source
--------------------
To compile from source under unix platforms, ``libffi`` is required. (For
Windows, this is already included with ``cffi``). ``libffi`` is available
through most package managers (e.g., ``yum install libffi-devel``, ``apt-get
install libffi-dev``, ``brew install libffi``). More information on installing
``libffi`` is available `here
<https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions>`__.

If doing a fresh checkout::

    $ git clone --recursive http://github.com/tgarc/pastream

If you already have a checkout::

    $ git submodule update --init

Then do a pip install from your working copy::

    $ pip install <path/to/checkout>


Building Documentation
----------------------
Documentation for pastream can be easily generated in a wide variety of formats
using Sphinx. Just follow the steps below.

Checkout the repository::

    $ git clone --recursive http://github.com/tgarc/pastream

Then use the included makefile/make.bat to generate documentation. (Here we
output to the html format)::

    $ cd pastream/docs
    $ make html


Examples
----------------
Record 1000 frames to file, then play it back:

.. code-block:: python

   import pastream as ps

   # Use *with* statements to auto-close the stream
   with ps.SoundFileInputStream('recording.wav') as stream:
       stream.frames = 1000
       stream.start()
       stream.wait() # Block until recording is done

   with ps.SoundFileOutputStream('recording.wav') as stream:
       stream.frames = 1000
       stream.start()
       stream.wait()

Grab (real) frequency transformed live audio stream with 50% overlap:

.. code-block:: python

   import pastream as ps, numpy as np

   chunksize = 1024
   window = np.hanning(chunksize)
   for x_l in ps.chunks(chunksize, overlap=chunksize//2, channels=1):
       X_l = np.fft.rfft(x_l * window)

See also the included examples under `/examples`.


Command Line Application
-----------------------------------
Once installed, the pastream application should be callable from your command
line. If you're familiar with `sox <http://sox.sourceforge.net/>`__ you'll
notice that some of the command line syntax is quite similar. Here are a few
examples to help get you started.

Display the help file::

    $ pastream -h

List available audio devices::

    $ pastream -l

Simultaneous play and record from the default audio device::

    $ pastream input.wav output.wav

Pipe input from sox using the AU format::

    $ sox -n -t au - synth sine 440 | pastream - output.wav

Play a RAW file::

    $ pastream -c1 -r48k -e=pcm_16 -o output.raw

Record 10 seconds of audio at 48kHz::

    $ pastream null output.wav -r48k -n=480k
