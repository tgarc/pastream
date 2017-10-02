#!/usr/bin/env python
# Copyright (c) 2017 Thomas J. Garcia
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
from __future__ import print_function as _print_function
try:                import Queue as _queue
except ImportError: import queue as _queue
import math as _math
import threading as _threading
import time as _time
import sys as _sys
import sounddevice as _sd
import soundfile as _sf
from _py_pastream import ffi as _ffi, lib as _lib
from pa_ringbuffer import _RingBufferBase
# defer importing numpy as it's fairly slow and not required for the CLI
_np = None


__version__ = '0.0.8'
__usage__ = "%(prog)s [options] input output"


# Set a default size for the audio callback ring buffer
_PA_BUFFERSIZE = 1 << 16

# Private states that determine how a stream completed
_FINISHED = 1
_ABORTED = 2
_STOPPED = 4
_INITIALIZED = 8

# Include xrun flags in nampespace
paInputOverflow = _lib.paInputOverflow
paInputUnderflow = _lib.paInputUnderflow
paOutputOverflow = _lib.paOutputOverflow
paOutputUnderflow = _lib.paOutputUnderflow


# create an independent module level default
default = type('default', (_sd.default.__class__,), {}).__call__()

def reset():
    _sd._terminate(); _sd._initialize()


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
            def __init__(self, ptr):
                self._ptr = ptr
                self._data = None
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
    def __init__(self, elementsize, buffer):
        self._ptr = self._ffi.new('PaUtilRingBuffer*')

        try:
            data = self._ffi.from_buffer(buffer)
        except TypeError:
            data = buffer

        size, rest = divmod(self._ffi.sizeof(data), elementsize)
        if rest:
            raise ValueError('buffer size must be multiple of elementsize')
        self._data = data

        falsesize = 1 << int(_math.ceil( _math.log(size, 2) )) if size else 0
        rcode = self._lib.PaUtil_InitializeRingBuffer(
            self._ptr, elementsize, falsesize, self._data)
        assert rcode == 0

        # we manually assign the buffersize; this will keep us from going over
        # the buffer bounds
        self._ptr.bufferSize = size


def _allocate_stream_buffer(stream, size, kind, atleast_2d=False, bufferclass=bytearray):
    if stream.isduplex:
        isoutput = kind == 'output'
        channels = stream.channels[isoutput]
        samplesize = stream.samplesize[isoutput]
        dtype = stream.dtype[isoutput]
    else:
        channels = stream.channels
        samplesize = stream.samplesize
        dtype = stream.dtype

    if issubclass(bufferclass, _RingBufferBase):
        return bufferclass(channels * samplesize, size)
    elif _np is None:
        return bufferclass(size * channels * samplesize)
    else:
        return _np.zeros((size, channels) if atleast_2d or channels > 1
                         else size * channels, dtype=dtype)


# In general, non-headerless soundfiles give us at least the number of channels
# and the samplerate. The sample encoding also directly translates to a
# PaStream dtype for formats that support PCM and FLOAT encodings. When the
# filetype does not support these encodings, the `subtype` must be passed
# manually.
#
# Notes:
#
# - playback file parameters passed will not be used when opening a soundfile
#   *except* for RAW/headerless formats
#
# - for duplex streams the playback file is used to set the stream
#   parameters. If an open recording file is passed, it will also be used to
#   set the output channels.
#
def _from_file(kind, recordfile=None, playbackfile=None, subtype=None, endian=None,
               format=None, **kwargs):
    """Create a stream using the charecteristics of a soundfile

    Other Parameters
    ----------------
    **kwargs
        Additional parameters to open Stream

    Returns
    -------
    stream
        Open stream object
    inp_fh
        Open recording file handler
    out_fh
        Open playback file handler

    """
    samplerate = kwargs.pop('samplerate', None)
    channels = kwargs.pop('channels', None)
    if kind == 'duplex':
        iformat, oformat = _sd._split(format)
        isubtype, osubtype = _sd._split(subtype)
        iendian, oendian = _sd._split(endian)
        ichannels, ochannels = _sd._split(channels)
    else:
        iformat = oformat = format
        isubtype = osubtype = subtype
        iendian = oendian = endian
        ichannels = ochannels = channels
    raw_output = oformat and oformat.lower() == 'raw' or False

    # Open the playback file and set channels/samplerate and dtype based on
    # it
    if playbackfile is not None:
        if not oformat and not raw_output:
            try:
                raw_output = \
                    getattr(playbackfile, 'name', playbackfile).rsplit('.', 1)[1].lower() == 'raw'
            except (AttributeError, IndexError):
                pass
        if isinstance(playbackfile, _sf.SoundFile):
            out_fh = playbackfile
        elif not raw_output:
            out_fh = _sf.SoundFile(playbackfile)
        elif not (samplerate and ochannels and osubtype):
            raise TypeError(
                "samplerate, channels, and subtype must be specified for RAW "
                "playback files")
        else:
            out_fh = _sf.SoundFile(playbackfile, 'r', samplerate, ochannels,
                                   osubtype, oendian, 'raw')

        # override samplerate and channels settings
        samplerate = out_fh.samplerate
        ochannels = out_fh.channels
    else:
        out_fh = playbackfile

    # If the recording file is already opened we can try to set stream
    # parameters based on it and confirm that it matches up with the
    # playback file in case of duplex streaming
    if isinstance(recordfile, _sf.SoundFile):
        if samplerate is None:
            samplerate = recordfile.samplerate
        elif recordfile.samplerate != samplerate:
            raise ValueError("Input and output file samplerates do not match!")
        # override input channels
        ichannels = recordfile.channels

    if kind == 'duplex':
        stream = DuplexStream(kwargs.pop('device', None), samplerate, (ichannels, ochannels), **kwargs)
    elif kind == 'input':
        stream = InputStream(kwargs.pop('device', None), samplerate, ichannels, **kwargs)
    else:
        stream = OutputStream(kwargs.pop('device', None), samplerate, ochannels, **kwargs)

    # If the recording file hasn't already been opened, we open it here using
    # the input file and stream settings, plus any user supplied arguments
    if recordfile is not None and not isinstance(recordfile, _sf.SoundFile):
        inp_fh = _soundfile_from_stream(stream, recordfile, mode='w',
            subtype=isubtype, format=iformat, endian=iendian)
    else:
        inp_fh = recordfile

    return stream, inp_fh, out_fh


# Open a soundfile based on stream parameters
# If an appropriate subtype cannot be found and none is specified, a TypeError
# is raised
def _soundfile_from_stream(stream, file, mode, **kwargs):
    # Try and determine the file extension here; we need to know if we
    # want to try and set a default subtype for the file
    fformat = kwargs.pop('format', None)
    if fformat is None:
        try:
            fformat = getattr(file, 'name', file).rsplit('.', 1)[1].lower()
        except (AttributeError, IndexError):
            fformat = None

    kindidx = 'r' in mode
    kind = 'output' if kindidx else 'input'
    if stream.isduplex:
        channels = stream.channels[kindidx]
        dtype = stream.dtype[kindidx]
        ssize = stream.samplesize[kindidx]
    else:
        dtype = stream.dtype
        ssize = stream.samplesize
        channels = stream.channels

    if kind == 'output':
        fformat = 'raw'

    subtype = kwargs.pop('subtype', None)
    endian = kwargs.get('endian', None)
    if not subtype:
        # For those file formats which support PCM or FLOAT, use the device
        # samplesize to make a guess at a default subtype
        if dtype == 'float32' and _sf.check_format(fformat, 'float', endian):
            subtype = 'float'
        else:
            subtype = 'pcm_{0}'.format(8 * ssize)
        if fformat and not _sf.check_format(fformat, subtype, endian):
            raise TypeError("Could not map stream datatype '{0}' to "
                "an appropriate subtype for '{1}' format; please specify"
                .format(dtype, fformat))

    return _sf.SoundFile(file, mode, int(stream.samplerate), channels,
                subtype=subtype, format=fformat, **kwargs)


# Default handler for writing input from a Stream to a SoundFile object
def _soundfilerecorder(stream, rxbuffer, inp_fh):
    if stream.isduplex:
        dtype = stream.dtype[0]
    else:
        dtype = stream.dtype

    chunksize = min(8192, len(rxbuffer))
    sleeptime = (chunksize - rxbuffer.read_available) / stream.samplerate
    if sleeptime > 0:
        _time.sleep(max(sleeptime, stream._offset / stream.samplerate))
    while not stream.aborted:
        # for thread safety, check the stream is active *before* reading
        active = stream.active
        frames = rxbuffer.read_available
        if frames == 0:
            # we've read everything and the stream is done; seeya!
            if not active: break
            # we're reading too fast, wait for a buffer write
            stream._rmisses += 1
            _time.sleep(0.0025)
            continue

        frames, buffregn1, buffregn2 = rxbuffer.get_read_buffers(frames)
        inp_fh.buffer_write(buffregn1, dtype=dtype)
        if len(buffregn2):
            inp_fh.buffer_write(buffregn2, dtype=dtype)
        rxbuffer.advance_read_index(frames)

        sleeptime = (chunksize - rxbuffer.read_available) / stream.samplerate
        if sleeptime > 0:
            _time.sleep(sleeptime)


# Default handler for reading input from a SoundFile object and writing it to a
# Stream
def _soundfileplayer(stream, txbuffer, out_fh, loop=False):
    readinto = out_fh.buffer_read_into
    if stream.isduplex:
        dtype = stream.dtype[1]
    else:
        dtype = stream.dtype

    chunksize = min(8192, len(txbuffer))
    sleeptime = (chunksize - txbuffer.write_available) / stream.samplerate
    if sleeptime > 0:
        _time.sleep(sleeptime)
    while not stream.finished:
        frames = txbuffer.write_available
        if not frames:
            stream._wmisses += 1
            _time.sleep(0.0025)
            continue

        frames, buffregn1, buffregn2 = txbuffer.get_write_buffers(frames)
        readframes = readinto(buffregn1, dtype=dtype)
        if len(buffregn2):
            readframes += readinto(buffregn2, dtype=dtype)

        if loop:
            # grab a memoryview to avoid copies
            buffregn1 = memoryview(buffregn1)
            buffregn2 = memoryview(buffregn2)
            while readframes < frames:
                out_fh.seek(0)
                readbytes = readframes * txbuffer.elementsize
                if readbytes < len(buffregn1):
                    readframes += readinto(buffregn1[readbytes:], dtype=dtype)
                else:
                    readframes += readinto(buffregn2[readbytes-len(buffregn1):], dtype=dtype)
        txbuffer.advance_write_index(readframes)
        if readframes < frames:
            break

        sleeptime = (chunksize - txbuffer.write_available) / stream.samplerate
        if sleeptime > 0:
            _time.sleep(sleeptime)


# TODO?: add option to do asynchronous exception raising
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
        if kwargs.get('callback', None) is None:
            # Init the C PyPaStream object
            self._cstream = _ffi.new("Py_PaStream*")
            _lib.init_stream(self._cstream)

            # Pass our data and callback to sounddevice
            kwargs['userdata'] = self._cstream
            kwargs['callback'] = _ffi.addressof(_lib, 'callback')
            kwargs['wrap_callback'] = None
            self.__frames = self._cstream.frames

        # These flags are used to tell when the callbacks have finished. We can
        # use them to abort writing of the ringbuffer.
        self.__statecond = _threading.Condition()
        self.__streamlock = _threading.RLock()
        self.__state = _INITIALIZED
        self.__aborting = False
        self.__exceptions = _queue.Queue()

        # Set up reader/writer threads
        self._owner_thread = _threading.current_thread()
        self._rxthread = self._txthread = None
        self._txthread_args = self._rxthread_args = None

        # TODO?: add support for C finished_callback function pointer
        self.__finished_callback = kwargs.pop('finished_callback', lambda x: None)

        def finished_callback():
            # Check for any errors that might've occurred in the callback
            msg = _ffi.string(self._cstream.errorMsg).decode('utf-8')
            if len(msg):
                exctype, excmsg = msg.split(':', 1) if ':' in msg else (msg, '')
                exctype = getattr(_sys.modules[__name__], exctype)
                self._set_exception(exctype(excmsg))

            with self.__statecond:
                # It's possible that the callback aborted itself so check if we
                # need to update our aborted flag here
                if self._cstream.last_callback == _sd._lib.paAbort \
                   or self.__aborting or not self.__exceptions.empty():
                    self.__state = _ABORTED | _FINISHED
                elif self._cstream.last_callback == _sd._lib.paComplete:
                    self.__state = _FINISHED
                else:
                    self.__state = _STOPPED | _FINISHED
                self.__aborting = False
                self.__statecond.notify_all()

            self.__finished_callback(self)

        super(Stream, self).__init__(kind, blocksize=blocksize,
            device=device, samplerate=samplerate, channels=channels,
            dtype=dtype, finished_callback=finished_callback, **kwargs)

        # DEBUG for measuring polling performance
        self._rmisses = self._wmisses = 0

        self._autoclose = False
        if kind == 'duplex':
            self._cstream.txElementSize = self.samplesize[1] * self.channels[1]
        elif kind == 'output':
            self._cstream.txElementSize = self.samplesize * self.channels

    def _set_thread(self, kind, target, buffersize=None, args=(), kwargs={}):
        if buffersize is None:
            buffersize = _PA_BUFFERSIZE

        if kind == 'output':
            buffer = self._txbuffer
            self._txthread_args = target, args, kwargs
        else:
            buffer = self._rxbuffer
            self._rxthread_args = target, args, kwargs

        # Only allocate a new buffer if an appropriate one is not already assigned
        if not isinstance(buffer, RingBuffer) or len(buffer) != buffersize:
            buffer = _allocate_stream_buffer(self, buffersize, kind, bufferclass=RingBuffer)

        # This will reset the PyPaStream buffer pointer in case it was
        # previously unset
        self._set_ringbuffer(kind, buffer)

    def _set_ringbuffer(self, kind, ringbuffer):
        if kind == 'input':
            self._cstream.rxbuffer = _ffi.cast('PaUtilRingBuffer*', ringbuffer._ptr)
            self._rxbuffer = ringbuffer
        else:
            self._cstream.txbuffer = _ffi.cast('PaUtilRingBuffer*', ringbuffer._ptr)
            self._txbuffer = ringbuffer

    def __stopiothreads(self):
        # !This function is *not* thread safe!
        currthread = _threading.current_thread()
        if self._rxthread is not None and self._rxthread.is_alive() \
           and self._rxthread != currthread:
            self._rxthread.join()
        if self._txthread is not None and self._txthread.is_alive() \
           and self._txthread != currthread:
            self._txthread.join()

    def _readwritewrapper(self, rwfunc, buff, *args, **kwargs):
        """\
        Wrapper for the reader and writer functions which acts as a kind
        of context manager.

        """
        try:
            rwfunc(self, buff, *args, **kwargs)
        except:
            # Defer the exception and delegate responsibility to the owner
            # thread
            self._set_exception()
            self.abort()

    def _reraise_exceptions(self):
        """\
        Raise the last deferred exception if one exists. If the caller's
        thread is not the stream owner this function does nothing.

        """
        currthread = _threading.current_thread()
        if currthread is self._rxthread or currthread is self._txthread:
            return

        try:
            exc = self.__exceptions.get(block=False)
        except _queue.Empty:
            return

        # To simplify things, we only care about the first exception raised
        self.__exceptions.queue.clear()

        if isinstance(exc, tuple):
            exctype, excval, exctb = exc
            if exctype is not None:
                excval = exctype(excval)
            if hasattr(excval, 'with_traceback'):
                raise excval.with_traceback(exctb)
            else:
                exec("raise excval, None, exctb")
        else:
            raise exc

    def _set_exception(self, exc=None):
        """\
        Queue an exception to be re-raised later using `_reraise_exceptions`.

        """
        try:
            self.__exceptions.put(exc or _sys.exc_info(), block=False)
        except _queue.Full:
            pass

    def wait(self, timeout=None):
        """\
        Block until stream state changes to finished/aborted/stopped or until
        the optional timeout occurs.

        Returns
        -------
        bool
            True unless the timeout occurs.

        """
        with self.__statecond:
            if self.__state == 0:
                self.__statecond.wait(timeout)
                self._reraise_exceptions()
            return self.__state > 0

    @property
    def isduplex(self):
        return hasattr(self.channels, '__len__')

    @property
    def aborted(self):
        """Check whether stream has been aborted.

        If True, it is guaranteed that the stream is in a finished state.

        """
        return self.__state & _ABORTED > 0

    @property
    def finished(self):
        """Check whether the stream is in a finished state.

        Will only be True if :meth:`start` has been called and the stream
        either completed sucessfully or was stopped/aborted.

        """
        return self.__state & _FINISHED > 0

    @property
    def status(self):
        """\
        The current PaStreamCallbackFlags status of the portaudio
        stream.

        """
        return self._cstream.status

    @property
    def xruns(self):
        return self._cstream.xruns

    @property
    def frame_count(self):
        """Running total of frames that have been processed.

        Each new starting of the stream resets this number to zero.

        """
        return self._cstream.frame_count

    @property
    def _offset(self):
        return self._cstream.offset

    @_offset.setter
    def _offset(self, value):
        self._cstream.offset = value

    @property
    def _pad(self):
        return self._cstream.pad

    @_pad.setter
    def _pad(self, value):
        # Note that the py_pastream callback doesn't act on 'pad' unless
        # frames < 0; thus, set 'frames' first to get deterministic behavior.
        frames = self.__frames
        if frames >= 0 and value >= 0:
            self._cstream.frames = value + frames
            # reset autoframes whenever we set frames from here
            self._cstream._autoframes = 0
        self._cstream.pad = value

    @property
    def _frames(self):
        # We fib a bit here: __frames is _cstream.frames minus any padding
        return self.__frames

    @_frames.setter
    def _frames(self, value):
        pad = self._cstream.pad
        if value > 0 and pad > 0:
            self._cstream.frames = value + pad
        else:
            self._cstream.frames = value
        self.__frames = value
        # reset autoframes whenever we set frames from here
        self._cstream._autoframes = 0

    def __enter__(self):
        return self

    def __exit__(self, exctype, excvalue, exctb):
        self.close()

    def _prepare(self):
        assert self.__state, "Stream has already been started!"

        # Apparently when using a PaStreamFinishedCallback the stream
        # *must* be stopped before starting the stream again or the
        # streamFinishedCallback will never be called
        if self.__state != _INITIALIZED and not self.stopped:
            super(Stream, self).stop()

        # Reset stream state machine
        with self.__statecond:
            self.__state = 0

        # Reset cstream info
        _lib.reset_stream(self._cstream)

        # Recreate the necessary threads
        if self._rxthread_args is not None:
            target, args, kwargs = self._rxthread_args
            self._rxthread = _threading.Thread(target=self._readwritewrapper,
                                               args=(target, self._rxbuffer) + args,
                                               kwargs=kwargs)
            self._rxthread.daemon = True
        else:
            self._rxthread = None

        if self._txthread_args is not None:
            target, args, kwargs = self._txthread_args
            self._txthread = _threading.Thread(target=self._readwritewrapper,
                                               args=(target, self._txbuffer) + args,
                                               kwargs=kwargs)
            self._txthread.daemon = True
        else:
            self._txthread = None

    # start is *not* thread safe! shouldn't have multiple callers anyway
    def start(self, prebuffer=True):
        """Start the audio stream

        Parameters
        ----------
        prebuffer : bool, optional
            For threading only: wait for the first output write before starting
            the audio stream. If not using threads this has no effect.
        """
        self._prepare()
        if self._txthread is not None:
            self._txthread.start()
            if prebuffer:
                while not _lib.PaUtil_GetRingBufferReadAvailable(self._cstream.txbuffer) \
                      and self._txthread.is_alive():
                    _time.sleep(0.0025)
            self._reraise_exceptions()
        super(Stream, self).start()

        if self._rxthread is not None:
            self._rxthread.start()
            self._reraise_exceptions()

    def stop(self):
        with self.__streamlock:
            super(Stream, self).stop()
        self.__stopiothreads()
        self._reraise_exceptions()

    def abort(self):
        with self.__statecond:
            self.__aborting = True
        with self.__streamlock:
            super(Stream, self).abort()
        self.__stopiothreads()
        self._reraise_exceptions()

    def close(self):
        # we take special care here to abort the stream first so that the
        # pastream pointer is still valid for the lifetime of the read/write
        # threads
        if not self.finished:
            with self.__statecond:
                self.__aborting = True
            with self.__streamlock:
                super(Stream, self).abort()
        self.__stopiothreads()
        with self.__streamlock:
            super(Stream, self).close()
        self._reraise_exceptions()

    def __repr__(self):
        if isinstance(self.device, int) or self.device[0] == self.device[1]:
            name = "'%s'" % _sd.query_devices(self._device)['name']
        else:
            name = tuple(_sd.query_devices(d)['name'] for d in self._device)
        if isinstance(self.channels, int) or self.channels[0] != self.channels[1]:
            channels = self.channels
        else:
            channels = self.channels[0]
        if self.dtype[0] == self.dtype[1]:
            # this is a hack that works only because there are no dtypes that
            # start with the same two characters
            dtype = self.dtype[0]
        else:
            dtype = self.dtype
        return ("{0.__name__}({1}, samplerate={2._samplerate:.0f}, "
                "channels={3}, dtype={4}, blocksize={2._blocksize})").format(
            self.__class__, name, self, channels, dtype)


# Mix-in purely for adding playback methods
class _OutputStreamMixin(object):

    def _set_playback(self, playback, loop=False, buffersize=None):
        if self.isduplex:
            channels = self.channels[1]
            elementsize = channels * self.samplesize[1]
        else:
            channels = self.channels
            elementsize = channels * self.samplesize

        if isinstance(playback, _sf.SoundFile):
            self._set_thread('output', _soundfileplayer, buffersize, args=(playback, loop))
            if playback.samplerate != self.samplerate or playback.channels != channels:
                raise ValueError("Playback file samplerate/channels mismatch")
            if loop and not playback.seekable():
                raise ValueError("Can't loop playback; file is not seekable")
            self._cstream.loop = 0
        else:
            try:
                data = _ffi.from_buffer(playback)
            except TypeError:
                data = playback
            frames = len(data) // elementsize

            playback = _LinearBuffer(elementsize, data)
            playback.advance_write_index(frames)
            self._set_ringbuffer('output', playback)
            self._cstream.loop = loop

            # null out any thread that was previously set
            self._txthread_args = None

        return playback

    def play(self, playback, frames=-1, pad=0, loop=False, blocking=False):
        """Play back audio data from a buffer or file

        Parameters
        -----------
        frames : int, optional
            Number of frames to play. (Note: This does *not* include the length of
            any additional padding). A negative value (the default) will cause the
            stream to continue until the send buffer is empty.
        pad : int, optional
            Number of zero frames to pad the playback with. A negative value causes
            padding to be automatically chosen so that the total playback length
            matches `frames` (or, if frames is negative, zero padding will be added
            indefinitely).

        """
        # Null out any rx thread or rx buffer pointer but note that we
        # intentionally do not clear the _rxbuffer in case that memory could be
        # used again
        self._rxthread_args = None
        self._cstream.rxbuffer = _ffi.NULL

        self._set_playback(playback, loop)

        self._pad = pad
        self._frames = frames

        self.start()
        if blocking:
            self.wait()


# Mix-in purely for adding recording methods
class _InputStreamMixin(object):

    def _set_capture(self, out, buffersize=None):
        if self.isduplex:
            channels = self.channels[0]
            elementsize = channels * self.samplesize[0]
        else:
            channels = self.channels
            elementsize = channels * self.samplesize

        if isinstance(out, _sf.SoundFile):
            self._set_thread('input', _soundfilerecorder, buffersize, args=(out,))
            if out.samplerate != self.samplerate or out.channels != channels:
                raise ValueError("Recording file samplerate/channels mismatch")
            return out
        elif isinstance(out, bytes):
            raise TypeError("out buffer type is read-only")

        buffer = _LinearBuffer(elementsize, out)
        self._set_ringbuffer('input', buffer)

        # null out any thread that was previously set
        self._rxthread_args = None

        return buffer

    def record(self, frames=None, offset=0, blocking=False, atleast_2d=False, out=None):
        """Record audio data to a buffer or file

        Parameters
        -----------
        frames : int, optional
            Number of frames to record. A negative value (the default) causes
            recordings to continue indefinitely.
        offset : int, optional
            Number of frames to discard from beginning of recording.

        Returns
        -------
        ndarray or bytearray or type(out)
            Recording buffer.

        """
        if frames is None and out is None:
            raise TypeError("at least one of {frames, out} is required")
        if out is None:
            out = _allocate_stream_buffer(self, frames - offset, 'input', atleast_2d)
        if atleast_2d and (_np is None or not isinstance(out, _np.ndarray)):
            raise ValueError("atleast_2d is only supported with numpy arrays")

        # Null out any previously set tx buffer/threads (see comment in play())
        self._txthread_args = None
        self._cstream.txbuffer = _ffi.NULL

        capture = self._set_capture(out)
        isbuffer = not isinstance(capture, _sf.SoundFile)
        if frames is not None:
            self._frames = frames
            if isbuffer:
                assert frames <= len(capture) + offset
        elif isbuffer:
            self._frames = len(capture) + offset
        else:
            self._frames = -1

        self._offset = offset

        self.start()
        if blocking:
            self.wait()

        return out

    #TODO add ability to pad out last chunk so it's the same length as the rest
    #TODO? if `out` is set, modify the chunksize to use the length of the buffer
    #TODO? when out is used have pypastream write to it directly, avoiding the copy
    def chunks(self, chunksize=None, overlap=0, frames=-1, pad=-1, offset=0,
               atleast_2d=False, playback=None, loop=False, out=None):
        """Read audio data in iterable chunks from a Portaudio stream.

        Similar in concept to PySoundFile library's
        :meth:`~soundfile.SoundFile.blocks` method. Returns an iterator over
        buffered audio chunks read from a Portaudio stream.  By default a
        direct view into the stream's ringbuffer is returned whenever
        possible. Setting an `out` buffer will of course incur an extra copy.

        Parameters
        ----------
        chunksize : int, optional
            Size of iterator chunks. If not specified the stream blocksize will
            be used. Note that if the blocksize is zero the yielded audio
            chunks may be of variable length depending on the audio backend.
        overlap : int, optional
            Number of frames to overlap across blocks.
        frames : int, optional
            Number of frames to play/record.
        pad : int, optional
            Playback padding. See :meth:`OutputStream.play`. Only applicable when playback is given.
        offset : int, optional
            Recording offset. See :meth:`InputStream.record`.
        atleast_2d : bool, optional
            Always return chunks as 2 dimensional arrays. Only valid when numpy
            is used.
        playback : buffer or SoundFile, optional
            Set playback audio. Only works for full duplex streams.
        loop : bool, optional
            Loop the playback audio.
        out : :class:`~numpy.ndarray` or buffer object, optional
            Alternative output buffer in which to store the result. Note that
            any buffer object - with the exception of :class:`~numpy.ndarray` - is
            expected to have single-byte elements as would be provided by e.g.,
            ``bytearray``. ``bytes`` objects are not recommended as they will
            incur extra copies (use ``bytearray`` instead).

        Yields
        ------
        :class:`~numpy.ndarray` or memoryview or cffi.buffer
            Buffer object with `chunksize` frames. If numpy is available
            defaults to :class:`~numpy.ndarray` otherwise a buffer of bytes is
            yielded (which is either a :class:`cffi.buffer` object or a
            ``memoryview``).

        """
        if self.isduplex:
            channels, ochannels = self.channels
            samplesize, osamplesize = self.samplesize
            latency = self.latency[0]
            dtype = self.dtype[0]
        else:
            latency = self.latency
            dtype = self.dtype
            channels = self.channels
            ochannels = samplesize = osamplesize = None

        if atleast_2d and (_np is None or not isinstance(out, _np.ndarray)):
            raise ValueError("atleast_2d is only supported with numpy arrays")

        varsize = False
        if not chunksize:
            if self.blocksize:
                chunksize = self.blocksize - overlap
            elif overlap:
                raise ValueError(
                    "Using overlap requires a non-zero chunksize or stream blocksize")
            else:
                varsize = True
                chunksize = int(round(latency * self.samplerate))

        if overlap >= chunksize:
            raise ValueError(
                "Overlap must be less than chunksize or stream blocksize")

        if playback is None:
            self._txthread_args = None
            self._cstream.txbuffer = _ffi.NULL
        elif not self.isduplex:
            raise ValueError("playback not supported; this stream is input only")
        else:
            self._set_playback(playback, loop)

        # Allocate a ringbuffer for double buffering input
        # Only allocate a new buffer if an appropriate one is not already assigned
        rxbuffer = self._rxbuffer
        if not isinstance(buffer, RingBuffer) or len(buffer) != _PA_BUFFERSIZE:
            rxbuffer = _allocate_stream_buffer(self, _PA_BUFFERSIZE, 'input', bufferclass=RingBuffer)
        self._set_ringbuffer('input', rxbuffer)

        numpy = False
        if out is not None:
            tempbuff = out
            if _np is not None and isinstance(out, _np.ndarray):
                numpy = True
                try:                   bytebuff = tempbuff.data.cast('B')
                except AttributeError: bytebuff = tempbuff.data
            else:
                bytebuff = tempbuff
        elif _np is None:
            if varsize: nbytes = len(rxbuffer) * rxbuffer.elementsize
            else:       nbytes = chunksize * rxbuffer.elementsize
            # Indexing into a bytearray creates a copy, so just use a
            # memoryview
            bytebuff = tempbuff = memoryview(bytearray(nbytes))
        else:
            numpy = True
            if channels > 1: atleast_2d = True
            if varsize: nframes = len(rxbuffer)
            else:       nframes = chunksize
            tempbuff = _np.zeros((nframes, channels) if atleast_2d
                                 else nframes * channels, dtype=dtype)
            try:                   bytebuff = tempbuff.data.cast('B')
            except AttributeError: bytebuff = tempbuff.data

        boverlap = overlap * rxbuffer.elementsize
        minframes = 1 if varsize else chunksize

        # fill the first overlap block with zeros
        if overlap:
            rxbuffer.write(bytearray(boverlap))

        # DEBUG
        # logf = open('chunks2.log', 'wt')
        # print("delta sleep lag yield misses frames", file=logf)
        # starttime = dt = rmisses = 0

        wait_time = leadtime = 0
        done = False
        starttime = _time.time()

        self._offset = offset
        self._frames = frames
        self._pad = pad
        self.start()

        try:
            sleeptime = latency - _time.time() + starttime - rxbuffer.read_available / self.samplerate
            if sleeptime > 0:
                _time.sleep(max(self._offset / self.samplerate, sleeptime))

            while not (self.aborted or done):
                # for thread safety, check the stream is active *before* reading
                active = self.active
                frames = rxbuffer.read_available
                lastTime = self._cstream.lastTime.currentTime
                if frames < minframes:
                    if not wait_time:
                        wait_time = self.time
                    if not active:
                        done = True
                        if frames == 0: break
                    else:
                        self._rmisses += 1
                        _time.sleep(0.0025)
                        continue
                elif wait_time:
                    leadtime = lastTime - wait_time
                    wait_time = 0

                # Debugging only
                # print("{0:f} {1:f} {2:f} {3:f} {4} {5}".format(
                #     1e3*(_time.time() - starttime), 1e3*sleeptime, 1e3*(self.time - lastTime),
                #     1e3*dt, self._rmisses - rmisses, frames - chunksize), file=logf)
                # rmisses = self._rmisses
                # starttime = _time.time()

                frames, buffregn1, buffregn2 = rxbuffer.get_read_buffers(
                    frames if varsize else chunksize)

                if out is not None or len(buffregn2):
                    buffsz1 = len(buffregn1)
                    bytebuff[:buffsz1] = buffregn1
                    bytebuff[buffsz1:buffsz1 + len(buffregn2)] = buffregn2
                    rxbuffer.advance_read_index(frames - overlap)
                    if numpy:
                        yield tempbuff[:frames]
                    else:
                        yield bytebuff[:frames * rxbuffer.elementsize]
                else:
                    if atleast_2d:
                        yield _np.frombuffer(buffregn1, dtype=dtype)\
                                 .reshape(frames, channels)
                    elif numpy:
                        yield _np.frombuffer(buffregn1, dtype=dtype)
                    else:
                        yield buffregn1
                    rxbuffer.advance_read_index(frames - overlap)

                # # DEBUG
                # dt = _time.time() - starttime

                sleeptime = (chunksize - rxbuffer.read_available) / self.samplerate \
                    + self._cstream.lastTime.currentTime - self.time              \
                    + leadtime
                if sleeptime > 0:
                    _time.sleep(sleeptime)
        except Exception:
            if _sd._initialized:
                self.abort()
            raise
        else:
            self._reraise_exceptions()
        finally:
            if _sd._initialized and self._autoclose:
                self.close()

class InputStream(_InputStreamMixin, Stream):
    """Record only stream.

    Parameters
    -----------
    frames : int, optional
        Number of frames to record. A negative value (the default) causes
        recordings to continue indefinitely.
    offset : int, optional
        Number of frames to discard from beginning of recording.
    reader : function, optional
        Dedicated function for reading from the input ring buffer.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to base stream class.

    """

    def __init__(self, *args, **kwargs):
        super(InputStream, self).__init__('input', *args, **kwargs)


class OutputStream(_OutputStreamMixin, Stream):
    """Playback only stream.

    writer : function, optional
        Dedicated function for feeding the output ring buffer.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to :class:`Stream`.

    """

    def __init__(self, *args, **kwargs):
        super(OutputStream, self).__init__('output', *args, **kwargs)


class DuplexStream(InputStream, OutputStream):
    """Full duplex audio streamer.

    Parameters
    -----------
    frames : int, optional
        Number of frames to play/record. (Note: This does *not* include the
        length of any additional padding).
    blocksize : int, optional
        Portaudio buffer size. If None or 0 (recommended), the Portaudio
        backend will automatically determine an optimal size.

    Other Parameters
    ----------------
    pad, writer
        See :class:`OutputStream`.
    offset, reader
        See :class:`InputStream`.
    device, channels, dtype, **kwargs
        Additional parameters to pass to :class:`Stream`.

    See Also
    --------
    :class:`OutputStream`, :class:`InputStream`

    """

    def __init__(self, *args, **kwargs):
        Stream.__init__(self, 'duplex', *args, **kwargs)

    def playrec(self, playback, frames=None, pad=0, offset=0, atleast_2d=False,
                loop=False, blocking=False, out=None):
        ichannels, ochannels = self.channels
        isamplesize, osamplesize = self.samplesize

        playback = self._set_playback(playback, loop)

        # if playback is a file with no determined length we could get a
        # nonsensical value which may be negative or a huge number; thus we
        # can't rely on it
        if frames is None and out is None and isinstance(playback, _sf.SoundFile):
            raise TypeError("at least one of {frames, out} is required when playback is a file")
        if out is None:
            # do we just ignore negative `pad` in this case? it doesn't really
            # make sense to use in this context
            # frames = 0 case?
            # frames < offset case?
            if frames is None:
                frames = len(playback)
            out = _allocate_stream_buffer(self, frames - offset + (pad if pad >= 0 else 0), 'input')

        if atleast_2d and (_np is None or not isinstance(out, _np.ndarray)):
            raise ValueError("atleast_2d is only supported with numpy arrays")

        capture = self._set_capture(out)
        isbuffer = not isinstance(capture, _sf.SoundFile)
        if frames is not None:
            self._frames = frames
            if isbuffer:
                assert frames + (pad if pad >= 0 else 0) <= len(capture) + offset
        elif isbuffer:
            self._frames = len(capture) + offset
        else:
            self._frames = -1

        self._offset = offset
        self._pad = pad

        self.start()
        if blocking:
            self.wait()

        return out


def chunks(chunksize=None, overlap=0, frames=-1, pad=0, offset=0,
           atleast_2d=False, playback=None, loop=False, out=None, **kwargs):
    """Read audio data in iterable chunks from a Portaudio stream.

    Parameters
    ------------
    chunksize, overlap, frames, pad, offset, atleast_2d, loop, out
        See :meth:`InputStream.chunks` for description.
    playback : :class:`~soundfile.SoundFile` compatible object, optional
        Optional playback file.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to :class:`Stream`.

    Yields
    -------
    ndarray or bytearray or type(out)
        buffer object with `chunksize` elements.

    See Also
    --------
    :meth:`InputStream.chunks`

    """
    if playback is not None:
        stream, null, playback_fh = _from_file('duplex', playbackfile=playback, **kwargs)
        _add_file_closer(stream, playback, playback_fh, None, None)
        playback = playback_fh
    else:
        stream = InputStream(**kwargs)
    stream._autoclose = True
    return stream.chunks(chunksize, overlap, frames, pad, offset, atleast_2d, playback, loop, out)


def _add_file_closer(stream, outf, out_fh, inpf, inp_fh):
    origclose = stream.close
    def close():
        try:
            origclose()
        finally:
            if not (outf is None or isinstance(outf, _sf.SoundFile)):
                out_fh.close()
            if not (inpf is None or isinstance(inpf, _sf.SoundFile)):
                inp_fh.close()
    stream.close = close


def fileplayer(playback, frames=-1, pad=0, loop=False, duplex=False, buffersize=None, **kwargs):
    stream, null, playback_fh = _from_file('duplex' if duplex else 'output', None, playback, **kwargs)

    stream._set_playback(playback_fh, loop, buffersize)

    stream._frames = frames
    stream._pad = pad

    _add_file_closer(stream, playback, playback_fh, None, None)

    return stream


def filerecorder(record, frames=-1, offset=0, duplex=False, buffersize=None, **kwargs):
    stream, record_fh, null = _from_file('duplex' if duplex else 'input', record, None, **kwargs)

    stream._set_capture(record_fh, buffersize)
    stream._frames = frames
    stream._offset = offset

    _add_file_closer(stream, None, None, record, record_fh)

    return stream


def fileplayrecorder(record, playback, frames=-1, pad=0, offset=0, loop=False, buffersize=None, **kwargs):
    stream, record_fh, playback_fh = _from_file('duplex', record, playback, **kwargs)

    stream._set_playback(playback_fh, loop, buffersize)
    stream._set_capture(record_fh, buffersize)

    stream._frames = frames
    stream._offset = offset
    stream._pad = pad

    _add_file_closer(stream, playback, playback_fh, record, record_fh)

    return stream


# Used solely for the pastream app
def _FileStreamFactory(record=None, playback=None, **kwargs):
    # allow user to specify 'frames' in seconds
    frames = kwargs.pop('frames', None)
    pad = kwargs.pop('pad', None)
    offset = kwargs.pop('offset', None)

    if record is not None and playback is not None:
        kind = 'duplex'
        stream = fileplayrecorder(record, playback, **kwargs)
        playback = stream._txthread_args[1][0]
        record = stream._rxthread_args[1][0]
    elif playback is not None:
        kwargs.pop('offset', None)
        kind = 'output'
        stream = fileplayer(playback, **kwargs)
        playback = stream._txthread_args[1][0]
    elif record is not None:
        kind = 'input'
        kwargs.pop('pad', None)
        kwargs.pop('loop', None)
        stream = filerecorder(record, **kwargs)
        record = stream._rxthread_args[1][0]
    else:
        raise ValueError("At least one of {playback, record} must be non-null.")

    # frames/pad/offset can all be specified in seconds (ie a multiple of samplerate)
    # so set them here after the stream is opened
    locs = locals()
    for k in ('frames', 'pad', 'offset'):
        v = locs[k]
        if v is None: continue
        if isinstance(v, str):
            v = int(round(float(v) * stream.samplerate))
        setattr(stream, '_' + k, v)

    return stream, record, playback, kind


def _get_parser(parser=None):
    import shlex
    from argparse import Action, ArgumentParser, RawDescriptionHelpFormatter

    if parser is None:
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                                add_help=False,
                                fromfile_prefix_chars='@', usage=__usage__,
                                description='''\
Cross platform audio playback and capture.''')
        parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()

    class ListStreamsAction(Action):
        def __call__(*args, **kwargs):
            print(_sd.query_devices())
            _sys.exit(0)

    def framestype(frames):
        if frames.endswith('s'):
            return sizetype(frames[:-1])
        return frames

    def dvctype(dvc):
        try:               return int(dvc)
        except ValueError: return dvc

    def sizetype(x):
        if   x.endswith('k'): x = int(float(x[:-1]) * 1e3)
        elif x.endswith('K'): x = int(float(x[:-1]) * 1024)
        elif x.endswith('m'): x = int(float(x[:-1]) * 1e6)
        elif x.endswith('M'): x = int(float(x[:-1]) * 1024 * 1024)
        else:                 x = int(float(x))
        return x

    def possizetype(x):
        x = sizetype(x)
        assert x > 0, "Must be a positive value."
        return x

    def nullortype(x, type=None):
        if not x or x.lower() == 'null':
            return None
        return x if type is None else type(x)

    def csvtype(arg, type=None):
        csvsplit = shlex.shlex(arg, posix=True)
        csvsplit.whitespace = ','; csvsplit.whitespace_split = True
        return tuple(map(type or str, csvsplit))

    parser.add_argument("input", type=nullortype,
        help='''\
Playback audio file. Use dash (-) to read from STDIN. Use 'null' or an empty
string ("") for record only.''')

    parser.add_argument("output", type=nullortype, nargs='?',
        help='''\
Output file for recording. Use dash (-) to write to STDOUT. Use 'null' or an
empty string ("") for playback only.''')

    genopts = parser.add_argument_group("general options")

    genopts.add_argument("-h", "--help", action="help",
        help="Show this help message and exit.")

    genopts.add_argument("-l", "--list", action=ListStreamsAction, nargs=0,
        help="List available audio device streams.")

    genopts.add_argument("-q", "--quiet", action='store_true',
        help="Don't print any status information.")

    genopts.add_argument("--version", action='version',
        version='%(prog)s ' + __version__, help="Print version and exit.")

    propts = parser.add_argument_group('''\
playback/record options. (size suffixes supported: k[ilo] K[ibi] m[ega] M[ebi])''')

    propts.add_argument("--buffersize", type=possizetype,
        default=_PA_BUFFERSIZE, help='''\
File buffering size in units of frames. Must be a power of 2. Determines the
maximum amount of buffering for the input/output file(s). Use higher values to
increase robustness against irregular file i/o behavior. Add a 'K' or 'M'
suffix to specify size in kibi or mebi units. (Default %(default)d)''')

    propts.add_argument("--loop", action='store_true', default=False,
        help="Loop playback indefinitely.")

    propts.add_argument("-d", "--duration", type=framestype, default=-1,
        help='''\
Limit playback/capture to a certain duration in
seconds. Alternatively, you may specify duration in samples by adding
an 's' suffix (e.g., 1ks == 1000 samples). If FRAMES is negative
(the default), then streaming will continue until there is no playback
data remaining or, if no playback was given, recording will continue
indefinitely.''')

    propts.add_argument("-o", "--offset", type=framestype, default=0, help='''\
Drop a number of frames from the start of a recording.''')

    propts.add_argument("-p", "--pad", type=framestype, nargs='?', default=0,
        const=-1, help='''\
Pad the input with frames of zeros. (Useful to avoid truncating full
duplex recording). If PAD is negative (the default if no argument is
given) then padding is chosen so that the total playback length
matches --duration. If frames is also negative, zero padding will be
added indefinitely.''')

    devopts = parser.add_argument_group("audio device options")

    devopts.add_argument("-b", "--blocksize", type=possizetype, help='''\
PortAudio buffer size in units of frames. If zero or not specified, backend
will decide an optimal size (recommended). ''')

    devopts.add_argument("-c", "--channels", metavar='channels[,channels]',
        type=lambda x: csvtype(x, lambda y: nullortype(y, int)),
        help="Number of input/output channels.")

    devopts.add_argument("-D", "--device", metavar='device[,device]',
        type=lambda x: csvtype(x, lambda y: nullortype(y, dvctype)),
        help='''\
Audio device name expression(s) or index number(s). Defaults to the
PortAudio default device(s).''')

    devopts.add_argument("-f", "--format", metavar="format[,format]",
        dest='dtype', type=lambda x: csvtype(x, nullortype),
        help='''\
Sample format(s) of audio device stream. Must be one of {%s}.'''
% ', '.join(['null'] + list(_sd._sampleformats.keys())))

    devopts.add_argument("-r", "--rate", dest='samplerate', type=possizetype,
        help='''\
Sample rate in Hz. Add a 'k' suffix to specify kHz.''')

    fileopts = parser.add_argument_group('''\
audio file formatting options. (options accept single values or pairs)''')

    fileopts.add_argument("-t", "--file_type", metavar="file_type[,file_type]",
        type=lambda x: csvtype(x, lambda y: nullortype(y, str.upper)),
        help='''\
Audio file type(s). (Required for RAW files). Typically this is determined
from the file header or extension, but it can be manually specified here. Must
be one of {%s}.''' % ', '.join(['null'] + list(_sf.available_formats().keys())))

    fileopts.add_argument("-e", "--encoding", metavar="encoding[,encoding]",
        type=lambda x: csvtype(x, lambda y: nullortype(y, str.upper)),
        help='''\
Sample format encoding(s). Note for output file encodings: for file types that
support PCM or FLOAT format, pastream will automatically choose the sample
format that most closely matches the output device stream; for other file
types, the subtype is required. Must be one of {%s}.'''
% ', '.join(['null'] + list(_sf.available_subtypes().keys())))

    fileopts.add_argument("--endian", metavar="endian[,endian]",
        type=lambda x: csvtype(x, lambda y: nullortype(y, str.lower)),
        help='''\
Sample endianness. Must be one of {%s}.''' % ', '.join(['null'] + ['file', 'big', 'little']))

    return parser


def _main(argv=None):
    import os, traceback

    if argv is None:
        argv = _sys.argv[1:]
    parser = _get_parser()
    args = parser.parse_args(argv)

    # Note that input/output from the cli perspective is reversed wrt the
    # pastream/portaudio library so we swap arguments here
    def unpack(x):
        return x[0] if x and len(x) == 1 else x and x[::-1]

    try:
        stream, record, playback, kind = _FileStreamFactory(
            args.output, args.input,
            buffersize=args.buffersize,
            loop=args.loop,
            offset=args.offset,
            pad=args.pad,
            frames=args.duration,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            endian=unpack(args.endian),
            subtype=unpack(args.encoding),
            format=unpack(args.file_type),
            device=unpack(args.device),
            channels=unpack(args.channels),
            dtype=unpack(args.dtype))
    except ValueError:
        traceback.print_exc()
        parser.print_usage()
        parser.exit(255)

    if args.output == '-' or args.quiet:
        _sys.stdout = open(os.devnull, 'w')

    statline = "\r{:8.3f}s ({:d} xruns, {:6.2f}% load)\r"
    print("<-", 'null' if playback is None else playback)
    print("->", 'null' if record is None else record)
    print(["--", "->", "<-"][['duplex', 'output', 'input'].index(kind)], stream)

    with stream:
        try:
            stream.start()
            t1 = _time.time()
            while stream.active:
                line = statline.format(_time.time() - t1, stream.xruns,
                                       100 * stream.cpu_load)
                _sys.stdout.write(line); _sys.stdout.flush()
                _time.sleep(0.15)
        except KeyboardInterrupt:
            stream.stop()
        finally:
            print()

    print("Callback info:")
    print("\tFrames processed: %d ( %7.3fs )"
          % (stream.frame_count, stream.frame_count / float(stream.samplerate)))
    print('''\
\txruns (under/over): input {0.inputUnderflows}/{0.inputOverflows}
\txruns (under/over): output {0.outputUnderflows}/{0.outputOverflows}'''
          .format(stream._cstream))
    print("\tWrite/read misses: %d/%d" % (stream._wmisses, stream._rmisses))

    return 0


if __name__ == '__main__':
    _sys.exit(_main())
else:
    try:
        import numpy as _np
    except ImportError:
        pass
