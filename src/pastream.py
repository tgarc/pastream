#!/usr/bin/env python
# Copyright (c) 2018 Thomas J. Garcia
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""pastream: GIL-less Portaudio Streams for Python
"""
from __future__ import print_function as _print_function, division as _division
import math as _math
import time as _time
import sys as _sys
import sounddevice as _sd
import soundfile as _sf
from _py_pastream import ffi as _ffi, lib as _lib
from pa_ringbuffer import _RingBufferBase


# For debugging
## from timeit import default_timer as timer
## gtime = None

__version__ = '0.2.0.post0'


# Include xrun flags in nampespace
paInputOverflow = _lib.paInputOverflow
paInputUnderflow = _lib.paInputUnderflow
paOutputOverflow = _lib.paOutputOverflow
paOutputUnderflow = _lib.paOutputUnderflow


class BufferFull(Exception):
    pass


class BufferEmpty(Exception):
    pass


class RingBuffer(_RingBufferBase):
    _ffi = _ffi
    _lib = _lib
    __doc__ = _RingBufferBase.__doc__

    @classmethod
    def _from_pointer(cls, pointer):
        class ProxyRingBuffer(cls):
            def __init__(self, ptr, data=None):
                self._ptr = ptr
                self._data = data
        return ProxyRingBuffer(pointer)


class _LinearBuffer(RingBuffer):
    """(internal) RingBuffer interface for non-power-of-2 sized buffers

    This class is essentially the same as RingBuffer but lies to the underlying
    pa_ringbuffer about its total size in order to allowing passing of
    arbitrary sized buffers to the py_pastream callback using the same
    RingBuffer interface.

    The big caveat using this class is that you *cannot* wrap write/read around
    the end of the buffer.

    """
    def __new__(cls, elementsize, buffer):
        try:
            data = _ffi.from_buffer(buffer)
        except TypeError:
            data = buffer

        size, rest = divmod(_ffi.sizeof(data), elementsize)
        if rest:
            raise ValueError('buffer size must be multiple of elementsize')

        falsesize = 1 << int(_math.ceil( _math.log(size, 2) )) if size else 0
        if falsesize == size:
            # This is a power of 2 buffer so just create a regular RingBuffer
            return RingBuffer(elementsize, buffer=data)

        self = super(_LinearBuffer, cls).__new__(cls)
        self._data = data
        self._ptr = _ffi.new('PaUtilRingBuffer*')

        rcode = _lib.PaUtil_InitializeRingBuffer(
            self._ptr, elementsize, falsesize, self._data)
        assert rcode == 0

        # we manually assign the buffersize; this will keep us from going over
        # the buffer bounds
        self._ptr.bufferSize = size

        return self

    def __init__(self, elementsize, buffer):
        pass

class Stream(_sd._StreamBase):
    """Base stream class from which all other stream classes derive.
    Note that this class inherits from :mod:`sounddevice`'s ``_StreamBase``
    class.
    """

    def __init__(self, kind, device=None, samplerate=None, channels=None,
                 dtype=None, blocksize=None, **kwargs):
        # Set up the C portaudio callback
        self._rxbuffer = self._txbuffer = None
        self._cstream = _ffi.NULL

        # Init the C PyPaStream object
        self._cstream = _ffi.new("Py_PaStream*")
        _lib.init_stream(self._cstream)

        # Pass our data and callback to sounddevice
        kwargs['userdata'] = self._cstream
        kwargs['callback'] = _ffi.addressof(_lib, 'callback')
        kwargs['wrap_callback'] = None

        super(Stream, self).__init__(kind, blocksize=blocksize,
            device=device, samplerate=samplerate, channels=channels,
            dtype=dtype, **kwargs)

        # DEBUG for measuring polling performance
        ## self._rmisses = self._wmisses = 0

        if kind == 'duplex':
            self._cstream.config.txElementSize = self.samplesize[1] * self.channels[1]
            self._cstream.config.rxElementSize = self.samplesize[0] * self.channels[0]
        elif kind == 'output':
            self._cstream.config.txElementSize = self.samplesize * self.channels
        else: #if kind == 'input':
            self._cstream.config.rxElementSize = self.samplesize * self.channels

    @property
    def isduplex(self):
        """Return whether this is a full duplex stream or not"""
        return hasattr(self.channels, '__len__')

    @property
    def xruns(self):
        """Running total of xruns.
        Each new starting of the stream resets this number to zero.
        """
        return self._cstream.stats.xruns

    @property
    def frame_count(self):
        """Running total of frames that have been processed.
        Each new starting of the stream resets this number to zero.
        """
        return self._cstream.stats.frame_count
        
    def start(self):
        # Ensure that previous stream is done
        self.abort()
        _lib.reset_stream(self._cstream)
        super(Stream, self).start()

    def reset(self):
        assert not self.active
        # Drop references to any buffers and external objects
        self._cstream.txbuffer = _ffi.cast('PaUtilRingBuffer*', _ffi.NULL)
        self._cstream.rxbuffer = _ffi.cast('PaUtilRingBuffer*', _ffi.NULL)
        self._rxbuffer = self._txbuffer = None
        _lib.reset_stream(self._cstream)

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exctype, excvalue, exctb):
        self.close()

    def close(self):
        super(Stream, self).close()
        self.unset_buffers()

    def __repr__(self):
        try:
            if self.device[0] == self.device[1]:
                name = "'%s'" % _sd.query_devices(self.device)['name']
            else:
                name = tuple(_sd.query_devices(d)['name'] for d in self.device)
        except TypeError:
            name = "'%s'" % _sd.query_devices(self.device)['name']
        try:
            if self.channels[0] != self.channels[1]:
                channels = self.channels
            else:
                channels = self.channels[0]
        except TypeError:
            channels = self.channels
        if self.dtype[0] == self.dtype[1]:
            # this is a hack that works only because there are no dtypes that
            # start with the same two characters
            dtype = self.dtype[0]
        else:
            dtype = self.dtype

        if _sys.version_info.major < 3:
            name = name.encode('utf-8', 'replace')

        return ("{0.__name__}({1}, samplerate={2._samplerate:.0f}, "
                "channels={3}, dtype='{4}', blocksize={2._blocksize})").format(
            self.__class__, name, self, channels, dtype)

def _main(argv=None):
    import sys
    import soundfile as sf

    ifile, ofile = sys.argv[1:3]

    if ifile == '-':
        ifile = None
    else:
        ibuf, ifs = sf.read(ifile)
    if ofile == '-':
        ofile = None
    else:
        obuf, ofs = sf.read(ofile)

    stream = Stream()
    with stream:
        try:
            stream.start()
            t1 = _time.time()
            while stream.active:
                dt = _time.time() - t1
                sys.stdout.write('.')
                _time.sleep(0.12)
        except KeyboardInterrupt:
            stream.stop()
        finally:
            print(file=sys.stdout)

    ## print(stream._rmisses, stream._wmisses, file=_sys.stderr)

    return 0



if __name__ == '__main__':
    _sys.exit(_main())
