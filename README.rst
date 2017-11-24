.. image:: https://badge.fury.io/py/pastream.svg
    :target: https://badge.fury.io/py/pastream

.. image:: https://travis-ci.org/tgarc/pastream.svg?branch=master
    :target: https://travis-ci.org/tgarc/pastream

.. image:: https://ci.appveyor.com/api/projects/status/wk52r5jy9ri7dsi9/branch/master?svg=true
    :target: https://ci.appveyor.com/project/tgarc/pastream/branch/master


GIL-less Portaudio Streams for Python
=====================================
`pastream` builds on top of `portaudio <http://www.portaudio.com/>`__ and the
excellent `sounddevice <http://github.com/spatialaudio/python-sounddevice>`__
python bindings to provide some more advanced functionality right out of the
box. Note that in addition to the pastream *library*, pastream includes a
`command line application`_ for playing
and recording audio files.

Documentation:
   http://pastream.readthedocs.io/

Source code repository and issue tracker:
   http://github.com/tgarc/pastream/


Features
========
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

Built-in support for working with SoundFiles and numpy ndarrays
    Seamless support for playback/recording of numpy ndarrays, generic buffer
    types, and SoundFiles.

Reader/Writer Threads
    pastream simplifies the process of implementing stream reader and writer
    threads to manipulate and/or generate data in the background while leaving
    the main thread free for higher level management tasks.


External Dependencies
=====================

There are a few compiled libraries pastream requires which *may* need to be
installed separately depending on your operating system. Windows users are
luckiest, they can skip this section entirely.

`libffi <https://sourceware.org/libffi/>`__ (Linux/Unix/MacOSX):
   Under Linux/Unix/MacOSX platforms you'll need to install the ffi
   library. (For Windows users, ffi is already included with the python cffi
   package.)  libffi is available through most package managers::

     $ yum install libffi-devel # Red-hat/CentOS Linux
     $ apt-get install libffi-dev # Ubuntu/debian derivatives
     $ brew install libffi # Homebrew on OSX

   More information on installing ``libffi`` is available in the cffi
   documentation `here
   <https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions>`__.

`PortAudio <http://www.portaudio.com>`__ and `libsndfile <http://www.mega-nerd.com/libsndfile/>`__ (Linux/Unix):
   Linux and Unix users will also need to install a recent version of the
   ``PortAudio`` and ``libsndfile`` libraries. (For Windows and OSX, the
   sounddevice and soundfile python packages include prebuilt versions for
   you.) You can either install the latest available from your package manager
   (e.g. ``apt-get install libportaudio2 libsndfile`` for debian/raspbian) or
   install the latest stable build from the package website (Recommended).


Installation
============
Once the above dependencies have been resolved, you can install pastream using
pip::

    $ pip install pastream


Building From Source
=====================
Clone pastream with the ``--recursive`` flag::

    $ git clone --recursive http://github.com/tgarc/pastream

Or, if you already have a checkout::

    $ cd <path/to/checkout>
    $ git submodule update --init

Finally, do a pip install from your local working copy::

    $ pip install <path/to/checkout>


Building Documentation
======================
Documentation for pastream can be easily generated in a wide variety of formats
using Sphinx. Just follow the steps below. 

Checkout the repository and cd into it::

    $ git clone http://github.com/tgarc/pastream
    $ cd pastream

Install documentation dependencies using requirements file::

    $ pip install -r docs/requirements.txt

Then use the included Makefile/make.bat to generate documentation. (Here we
output to the html format)::

    $ cd docs
    $ make html


Examples
========
Record one second of audio to memory, then play it back:

.. code-block:: python

   import pastream as ps

   # Use *with* statements to auto-close the stream
   with ps.DuplexStream() as stream:
       out = stream.record(int(stream.samplerate), blocking=True)
       stream.play(out, blocking=True)

Playback 10 seconds of a file, adding zero padding if the file is shorter, and
record the result to memory:

.. code-block:: python

   import pastream as ps, soundfile as sf

   with sf.SoundFile('my-file.wav') as infile, ps.DuplexStream.from_file(infile) as stream:
       out = stream.playrec(infile, frames=10 * int(stream.samplerate), pad=-1, blocking=True)

Grab (real) frequency transformed live audio stream with 50% overlap:

.. code-block:: python

   import pastream as ps, numpy as np

   chunksize = 1024
   window = np.hanning(chunksize)
   for x_l in ps.chunks(chunksize, overlap=chunksize//2, channels=1):
       X_l = np.fft.rfft(x_l * window)

Generate a pure tone on-the-fly

.. code-block:: python

   import time
   import pastream as ps
   import numpy as np

   # A simple tone generator
   def tone_generator(stream, buffer, f, loop=False):
       fs = stream.samplerate

       # Create a time index
       t = 2*np.pi*f*np.arange(len(buffer), dtype=stream.dtype) / fs

       # Loop until the stream stops
       while not stream.finished:
           frames = buffer.write_available
           if not frames:
               time.sleep(0.010)
               continue

           # Get the write buffers directly to avoid making any extra copies
           frames, part1, part2 = buffer.get_write_buffers(frames)

           out = np.frombuffer(part1, dtype=stream.dtype)
           np.sin(t[:len(out)], out=out)

           if len(part2):
               # part2 will be nonempty whenever we wrap around the end of the ring buffer
               out = np.frombuffer(part2, dtype=stream.dtype)
               np.sin(t[:len(out)], out=out)

           # flag that we've added data to the buffer
           buffer.advance_write_index(frames)

           # advance the time index
           t += 2*np.pi*f*frames / fs

   with ps.OutputStream(channels=1) as stream:
       # Set our tone generator as the source and pass along the frequency
       freq = 1000
       stream.set_source(tone_generator, args=(freq,))

       # Busy-wait to allow for keyboard interrupt
       stream.start()
       while stream.active:
           time.sleep(0.1)

See also the included examples under `/examples`.


Command Line Application
========================
Once installed, the pastream application should be callable from your command
line. If you're familiar with `SoX <http://sox.sourceforge.net/>`__ you'll
notice that some of the command line syntax is quite similar. Here are a few
examples to help get you started.

Display the help file::

    $ pastream -h

List available audio devices::

    $ pastream -l

Simultaneous play and record from the default audio device::

    $ pastream input.wav output.wav

Pipe input from sox using the AU format and record the playback::

    $ sox -n -t au - synth sine 440 | pastream - output.wav

Play a RAW file::

    $ pastream -c1 -r48k -e=pcm_16 output.raw

Record 10 minutes of audio at 48kHz::

    $ pastream null output.wav -r48k -d10:00
