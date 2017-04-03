pastream Portaudio Streams for Python
======================================

`pastream` builds on top of `portaudio <http://www.portaudio.com/>` and the
excellent `sounddevice <http://github.com/spatialaudio/sounddevice>` python
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

- `sounddevice <http://github.com/spatialaudio/sounddevice>`

- `soundfile <https://github.com/bastibe/PySoundFile`

- (Optional) `numpy <http://www.numpy.org/>`


Installation
------------

If doing a fresh checkout:

    $ git clone --recursive http://github.com/tgarc/pastream

If you already have a checkout:

    $ git submodule update --init

Then do a pip install:

    $ pip install pastream
