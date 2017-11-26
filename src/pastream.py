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


# For debugging
## from timeit import default_timer as timer
## gtime = None

__version__ = '0.1.2'


# Set a default size for the audio callback ring buffer
_PA_BUFFERSIZE = 1 << 16

# Determines the blocksize for reading/writing sound files
_FILECHUNKSIZE = 4096
_MAXCHUNKSIZE = 1 << 18

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


# Round up to the closest multiple of unit
def _unitceil(x, unit):
    return unit * ((x + unit - 1) // unit)


# Default handler for writing input from a Stream to a SoundFile object
def _soundfilerecorder(stream, rxbuffer, inp_fh):
    try:
        latency = stream.latency[0]
        dtype = stream.dtype[0]
    except TypeError:
        latency = stream.latency
        dtype = stream.dtype

    ## global gtime

    periodsize = max(int(round(latency * stream.samplerate)), stream.blocksize)
    maxframes = _MAXCHUNKSIZE // rxbuffer.elementsize

    # Set chunksize to a multiple of _FILECHUNKSIZE
    chunksize = _unitceil(periodsize + _FILECHUNKSIZE, periodsize)
    chunksize = min(len(rxbuffer), maxframes, chunksize)

    sleeptime = (chunksize - rxbuffer.read_available + stream._offset) / stream.samplerate
    if sleeptime > 0:
        _time.sleep(sleeptime)

    sleeptime = max(chunksize - periodsize // 8, periodsize) / stream.samplerate
    ## _sys.stderr.write('r:%d r:%d ' % (periodsize, max(chunksize - periodsize // 8, periodsize)))
    while not stream.aborted:
        # for thread safety, check the stream is active *before* reading
        active = stream.active
        frames = min(rxbuffer.read_available, maxframes)
        if frames == 0:
            # we've read everything and the stream is done; seeya!
            if not active:
                break
            ## stream._rmisses += 1
            ## _time.sleep(latency)
            ## continue

        ## print('1', timer() - gtime, frames)

        frames, buffregn1, buffregn2 = rxbuffer.get_read_buffers(frames)
        inp_fh.buffer_write(buffregn1, dtype=dtype)
        if len(buffregn2):
            inp_fh.buffer_write(buffregn2, dtype=dtype)

        rxbuffer.advance_read_index(frames)
        _time.sleep(sleeptime)


# Default handler for reading input from a SoundFile object and
# writing it to a Stream
def _soundfileplayer(stream, txbuffer, out_fh, loop=False):
    try:
        latency = stream.latency[1]
        dtype = stream.dtype[1]
    except TypeError:
        latency = stream.latency
        dtype = stream.dtype

    ## global gtime

    periodsize = max(int(round(latency * stream.samplerate)), stream.blocksize)
    maxframes = _MAXCHUNKSIZE // txbuffer.elementsize

    # Set chunksize to a multiple of _FILECHUNKSIZE
    chunksize = _unitceil(periodsize + _FILECHUNKSIZE, periodsize)
    chunksize = min(len(txbuffer), maxframes, chunksize)
    sleeptime = max(chunksize - periodsize // 8, periodsize) / stream.samplerate

    readinto = out_fh.buffer_read_into
    ## _sys.stderr.write('w:%d w:%d ' % (periodsize, max(chunksize - periodsize // 8, periodsize)))
    while not stream.finished:
        frames = min(txbuffer.write_available, maxframes)

        ## if frames == 0:
        ##    stream._wmisses += 1
        ## print('0', timer() - gtime, frames)

        frames, buffregn1, buffregn2 = txbuffer.get_write_buffers(txbuffer.write_available)
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

        _time.sleep(sleeptime)


# TODO?: add option to do asynchronous exception raising
class Stream(_sd._StreamBase):
    """Base stream class from which all other stream classes derive.

    Note that this class inherits from :mod:`sounddevice`'s ``_StreamBase``
    class.

    """
    _soundfileplayer = staticmethod(_soundfileplayer)
    _soundfilerecorder = staticmethod(_soundfilerecorder)

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
        self._rxthread = self._txthread = None
        self._txthread_args = self._rxthread_args = None

        # TODO: add support for C finished_callback function pointer
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
        ## self._rmisses = self._wmisses = 0

        self._autoclose = False
        if kind == 'duplex':
            self._cstream.txElementSize = self.samplesize[1] * self.channels[1]
        elif kind == 'output':
            self._cstream.txElementSize = self.samplesize * self.channels

    def __stopiothreads(self):
        # !This function is *not* thread safe!
        currthread = _threading.current_thread()
        if self._rxthread is not None and self._rxthread.is_alive() \
           and self._rxthread != currthread:
            self._rxthread.join()
        if self._txthread is not None and self._txthread.is_alive() \
           and self._txthread != currthread:
            self._txthread.join()

    # TODO: add ability to re-enter rwfunc without having to recreate the thread
    def _readwritewrapper(self, rwfunc, *args, **kwargs):
        """\
        Wrapper for the reader and writer functions which acts as a kind
        of context manager.

        """
        try:
            rwfunc(self, *args, **kwargs)
        except:
            # Defer the exception and delegate to the owner thread
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

    def _allocate_buffer(self, size, kind, atleast_2d=False, bufferclass=bytearray):
        isoutput = kind == 'output'
        try:
            channels = self.channels[isoutput]
            samplesize = self.samplesize[isoutput]
            dtype = self.dtype[isoutput]
        except TypeError:
            channels = self.channels
            samplesize = self.samplesize
            dtype = self.dtype

        if issubclass(bufferclass, _RingBufferBase):
            return bufferclass(channels * samplesize, size)
        try:
            import numpy
            return numpy.zeros((size, channels) if atleast_2d or channels > 1
                               else size * channels, dtype=dtype)
        except ImportError:
            return bufferclass(size * channels * samplesize)

    @classmethod
    def from_file(cls, file, *args, **kwargs):
        """Create a stream using the charecteristics of a soundfile

        Parameters
        ----------
        file : SoundFile or str or int or file-like object

        Other Parameters
        ----------------
        *args, **kwargs
            Arguments to pass to Stream constructor

        Returns
        -------
        Stream or Stream subclass instance
            Open stream

        See Also
        --------
        :meth:`InputStream.to_file`

        """
        if not isinstance(file, _sf.SoundFile):
            file = _sf.SoundFile(file)

        if kwargs.get('samplerate', None) is None:
            kwargs['samplerate'] = file.samplerate

        if kwargs.get('channels', None) is None:
            kwargs['channels'] = file.channels

        return cls(*args, **kwargs)

    def wait(self, timeout=None):
        """Block until stream state changes to finished/aborted/stopped or until the
        optional timeout occurs.

        Parameters
        ----------
        time : float, optional
            Optional timeout in seconds.

        Returns
        -------
        bool
            True unless the timeout occurs.

        """
        with self.__statecond:
            if self.__state == 0:
                self.__statecond.wait(timeout)
                if self.__state:
                    # make sure any reader/writer threads are done before returning!
                    # TODO: need to think this through for corner cases...
                    self.__stopiothreads()
                self._reraise_exceptions()
            return self.__state > 0

    @property
    def isduplex(self):
        """Return whether this is a full duplex stream or not"""
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
                                               args=(target,) + args,
                                               kwargs=kwargs)
            self._rxthread.daemon = True
        else:
            self._rxthread = None

        if self._txthread_args is not None:
            target, args, kwargs = self._txthread_args
            self._txthread = _threading.Thread(target=self._readwritewrapper,
                                               args=(target,) + args,
                                               kwargs=kwargs)
            self._txthread.daemon = True
        else:
            self._txthread = None

    # start is *not* thread safe! shouldn't have multiple callers anyway
    def start(self, prebuffer=True):
        """Start the audio stream

        Parameters
        ----------
        prebuffer : bool or int, optional
            Wait for a number of frames to be written to the output
            buffer before starting the audio stream. If True is given
            just wait for the first write. If not using threads or the
            stream is not an output stream this has no effect.
        """
        self._prepare()

        ## global gtime
        ## gtime = timer()

        if self._txthread is not None:
            self._txthread.start()
            txbuffer = self._cstream.txbuffer
            prebuffer = int(prebuffer)
            while _lib.PaUtil_GetRingBufferReadAvailable(txbuffer) < prebuffer\
                and self._txthread.is_alive():
                _time.sleep(0.0025)
            self._reraise_exceptions()

        with self.__streamlock:
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
        # we take special care here to abort the stream first so that
        # the pastream pointer is still valid for the lifetime of the
        # read/write threads
        if not self.finished:
            with self.__statecond:
                self.__aborting = True
            with self.__streamlock:
                super(Stream, self).abort()

        self.__stopiothreads()

        with self.__streamlock:
            super(Stream, self).close()

        # Drop references to any buffers and external objects
        self._txthread_args = self._rxthread_args = None
        self._rxbuffer = self._txbuffer = None

        self._reraise_exceptions()

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


# Mix-in purely for adding playback methods
class _OutputStreamMixin(object):

    def set_source(self, source, buffersize=None, loop=False, args=(), kwargs={}):
        '''Set the playback source for the audio stream

        Parameters
        -----------
        source : function or RingBuffer or SoundFile or buffer type
            Playback source. If `source` is a function it must be of the form:
            ``function(stream, ringbuffer, *args, loop=<bool>,**kwargs)``.
            Funcion sources are useful if you want to handle generating
            playback in some custom way. For example, `source` could be a
            function that reads audio data from a socket. This function will be
            called from a separate thread whenever the stream is started and is
            expected to close itself whenever the stream becomes inactive. For
            an example see the ``_soundfileplayer`` function in the source code
            for this module.
        loop : bool, optional
            Whether to enable playback looping.
        buffersize : int, optional
            RingBuffer size to use for double buffering audio data. Only
            applicable if `source` is a function or SoundFile. Must be a power
            of 2.

        Other Parameters
        -----------------
        args, kwargs
            Additional arguments to pass if `source` is a function.

        Returns
        -------
        RingBuffer instance
            RingBuffer wrapper interface from which audio device will read audio data.

        See Also
        --------
        :meth:`InputStream.set_sink`

        '''
        try:
            channels = self.channels[1]
            elementsize = channels * self.samplesize[1]
        except TypeError:
            channels = self.channels
            elementsize = channels * self.samplesize

        if buffersize is None:
            buffersize = _PA_BUFFERSIZE

        writer = None
        if isinstance(source, _sf.SoundFile):
            writer = self._soundfileplayer
            if source.samplerate != self.samplerate or source.channels != channels:
                raise ValueError("Playback file samplerate/channels mismatch")
            if loop and not source.seekable():
                raise ValueError("Can't loop playback; file is not seekable")
            args, kwargs = (source,), {}
        elif isinstance(source, RingBuffer):
            buffer = source
        elif callable(source):
            writer = source
        else:
            buffer = _LinearBuffer(elementsize, source)
            buffer.advance_write_index(len(buffer))

        if writer is not None:
            # Only allocate a new buffer if an appropriate one is not already assigned
            buffer = self._txbuffer
            if buffer is None or len(buffer) != buffersize:
                buffer = RingBuffer(elementsize, buffersize)
            else:
                buffer.flush()

            # Assume the writer will take care of looping
            self._cstream.loop = 0
            kwargs['loop'] = loop
            self._txthread_args = writer, (buffer,) + args, kwargs
        else:
            self._cstream.loop = loop
            # null out any thread that was previously set
            self._txthread_args = None

        self._cstream.txbuffer = _ffi.cast('PaUtilRingBuffer*', buffer._ptr)
        self._txbuffer = buffer

        return buffer

    def play(self, playback, frames=-1, pad=0, loop=False, buffersize=None, blocking=False):
        """Play back audio data from a buffer or file

        Parameters
        -----------
        playback : buffer or SoundFile
            Playback source.
        frames : int, optional
            Number of frames to play. (Note: This does *not* include the length of
            any additional padding). A negative value (the default) will cause the
            stream to continue until the send buffer is empty.
        pad : int, optional
            Number of zero frames to pad the playback with. A negative value causes
            padding to be automatically chosen so that the total playback length
            matches `frames` (or, if frames is negative, zero padding will be added
            indefinitely).
        buffersize : int
            Buffer size to use for (double) buffering audio data from
            file. Only applicable when `playback` is a file. Must be a power of
            2.

        """
        # Null out any rx thread or rx buffer pointer but note that we
        # intentionally do not clear the _rxbuffer in case that memory could be
        # used again
        self._rxthread_args = None
        self._cstream.rxbuffer = _ffi.NULL

        self.set_source(playback, buffersize, loop)

        self._pad = pad
        self._frames = frames

        self.start()
        if blocking:
            self.wait()


# Mix-in purely for adding recording methods
class _InputStreamMixin(object):

    def to_file(self, file, mode='w', **kwargs):
        '''Open a SoundFile for writing based on stream characteristics

        Parameters
        ----------
        file : str or int or file-like object
            File to open as SoundFile
        **kwargs
            Additional arguments to pass to SoundFile constructor

        Raises
        ------
        TypeError
            If no subtype was given and an appropriate subtype could
            not be guessed.

        Returns
        -------
        SoundFile

        See Also
        --------
        :meth:`Stream.from_file`

        '''
        # Try and determine the file extension here; we need to know if we
        # want to try and set a default subtype for the file
        fformat = kwargs.pop('format', None)
        if fformat is None:
            try:
                fformat = getattr(file, 'name', file).rsplit('.', 1)[1].lower()
            except (AttributeError, IndexError):
                fformat = None

        try:
            channels = self.channels[0]
            dtype = self.dtype[0]
            ssize = self.samplesize[0]
        except TypeError:
            channels = self.channels
            dtype = self.dtype
            ssize = self.samplesize

        if kwargs.get('samplerate', None) is None:
            kwargs['samplerate'] = int(self.samplerate)
        if kwargs.get('channels', None) is None:
            kwargs['channels'] = channels

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

        return _sf.SoundFile(file, mode, subtype=subtype, format=fformat, **kwargs)

    def set_sink(self, sink, buffersize=None, args=(), kwargs={}):
        '''Set the recording sink for the audio stream

        Parameters
        -----------
        sink : function or RingBuffer or SoundFile or buffer type
            Recording sink. If `sink` is a function it must be of the form:
            ``function(stream, ringbuffer, *args, **kwargs)``. Funcion sources
            are useful if you want to handle capturing of audio data in some
            custom way. For example, `sink` could be a function that writes
            audio data directly to a socket.  This function will be called from
            a separate thread whenever the stream is started and is expected to
            close itself whenever the stream becomes inactive. For an example
            see the ``_soundfilerecorder`` function in the source code for this
            module.
        buffersize : int, optional
            RingBuffer size to use for (double) buffering audio data. Only
            applicable when `sink` is either a file or function. Must be a
            power of 2.
        args, kwargs
            Additional arguments to pass if `sink` is a function.

        Returns
        -------
        RingBuffer instance
            RingBuffer wrapper interface to which audio device will write audio data.

        See Also
        --------
        :meth:`OutputStream.set_source`

        '''
        try:
            channels = self.channels[0]
            elementsize = channels * self.samplesize[0]
        except TypeError:
            channels = self.channels
            elementsize = channels * self.samplesize

        if buffersize is None:
            buffersize = _PA_BUFFERSIZE

        reader = None
        if isinstance(sink, _sf.SoundFile):
            reader = self._soundfilerecorder
            if sink.samplerate != self.samplerate or sink.channels != channels:
                raise ValueError("Recording file samplerate/channels mismatch")
            args, kwargs = (sink,) + args, {}
        elif isinstance(sink, RingBuffer):
            buffer = sink
        elif callable(sink):
            reader = sink
        else:
            buffer = _LinearBuffer(elementsize, sink)

        if reader is not None:
            # Only allocate a new buffer if an appropriate one is not already assigned
            buffer = self._rxbuffer
            if buffer is None or len(buffer) != buffersize:
                buffer = RingBuffer(elementsize, buffersize)
            else:
                buffer.flush()

            self._rxthread_args = reader, (buffer,) + args, kwargs
        else:
            # null out any thread that was previously set
            self._rxthread_args = None

        self._cstream.rxbuffer = _ffi.cast('PaUtilRingBuffer*', buffer._ptr)
        self._rxbuffer = buffer

        return buffer

    def record(self, frames=None, offset=0, atleast_2d=False, buffersize=None, blocking=False, out=None):
        """Record audio data to a buffer or file

        Parameters
        -----------
        frames : int, sometimes optional
            Number of frames to record. Can be omitted if `out` is specified.
        offset : int, optional
            Number of frames to discard from beginning of recording.
        buffersize : int, optional
            Buffer size to use for (double) buffering audio data to file. Only
            applicable when `out` is a file. Must be a power of 2.
        out : buffer or SoundFile, optional
            Output sink.

        Returns
        -------
        ndarray or bytearray or type(out)
            Recording destination.

        See Also
        --------
        :meth:`OutputStream.play`, :meth:`DuplexStream.playrec`

        """
        try:
            import numpy
        except ImportError:
            numpy = None

        if frames is None and out is None:
            raise TypeError("at least one of {frames, out} is required")
        if out is None:
            out = self._allocate_buffer(frames - offset, 'input', atleast_2d)
        if atleast_2d and (numpy is None or not isinstance(out, numpy.ndarray)):
            raise ValueError("atleast_2d is only supported with numpy arrays")

        # Null out any previously set tx buffer/threads (see comment in play())
        self._txthread_args = None
        self._cstream.txbuffer = _ffi.NULL

        capture = self.set_sink(out, buffersize)
        isbuffer = not isinstance(out, _sf.SoundFile)
        if frames is not None:
            self._frames = frames
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
               atleast_2d=False, playback=None, loop=False, buffersize=None,
               out=None):
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
            Playback padding. See :meth:`OutputStream.play`. Only applicable
            when playback is given.
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
            any buffer object - with the exception of :class:`~numpy.ndarray` -
            is expected to have single-byte elements as would be provided by
            e.g., ``bytearray``.

        Yields
        ------
        :class:`~numpy.ndarray` or memoryview or cffi.buffer
            Buffer object with `chunksize` frames. If numpy is available
            defaults to :class:`~numpy.ndarray` otherwise a buffer of bytes is
            yielded (which is either a :class:`cffi.buffer` object or a
            ``memoryview``).

        See Also
        --------
        :meth:`chunks`

        """
        try:
            channels = self.channels[0]
            samplesize = self.samplesize[0]
            dtype = self.dtype[0]
            latency = self.latency[0]
        except TypeError:
            channels = self.channels
            samplesize = self.samplesize
            dtype = self.dtype
            latency = self.latency
        elementsize = channels * samplesize

        try:
            import numpy
        except ImportError:
            numpy = None

        if atleast_2d and (numpy is None or not isinstance(out, numpy.ndarray)):
            raise ValueError("atleast_2d is only supported with numpy arrays")

        periodsize = int(round(latency * self.samplerate))
        varsize = False
        if not chunksize:
            if self.blocksize:
                chunksize = self.blocksize - overlap
            elif overlap:
                raise ValueError(
                    "Using overlap requires a non-zero chunksize or stream blocksize")
            else:
                varsize = True
                chunksize = periodsize

        if overlap >= chunksize:
            raise ValueError(
                "Overlap must be less than chunksize or stream blocksize")

        if buffersize is None:
            buffersize = _PA_BUFFERSIZE

        # Allocate a ringbuffer for double buffering input
        rxbuffer = self._rxbuffer
        if not isinstance(rxbuffer, RingBuffer) or len(rxbuffer) != buffersize:
            rxbuffer = RingBuffer(elementsize, buffersize)
        else:
            rxbuffer.flush()
        self._cstream.rxbuffer = _ffi.cast('PaUtilRingBuffer*', rxbuffer._ptr)
        self._rxbuffer = rxbuffer

        # Clear any previous receive thread
        self._rxthread_args = None

        if playback is None:
            self._txthread_args = None
            self._cstream.txbuffer = _ffi.NULL
        elif not self.isduplex:
            raise ValueError("playback not supported; this stream is input only")
        elif playback is True:
            pass
        else:
            self.set_source(playback, buffersize, loop)

        ndarray = False
        if out is not None:
            tempbuff = out
            if numpy is not None and isinstance(out, numpy.ndarray):
                ndarray = True
                try:                   bytebuff = tempbuff.data.cast('B')
                except AttributeError: bytebuff = tempbuff.data
            else:
                bytebuff = tempbuff
        elif numpy is None:
            if varsize: nbytes = len(rxbuffer) * elementsize
            else:       nbytes = chunksize * elementsize
            # Indexing into a bytearray creates a copy, so wrap it with a
            # memoryview
            bytebuff = tempbuff = memoryview(bytearray(nbytes))
        else:
            ndarray = True
            if varsize: nframes = len(rxbuffer)
            else:       nframes = chunksize
            if channels > 1: atleast_2d = True
            tempbuff = numpy.zeros((nframes, channels) if atleast_2d
                                   else nframes * channels, dtype=dtype)
            try:                   bytebuff = tempbuff.data.cast('B')
            except AttributeError: bytebuff = tempbuff.data

        # fill the first overlap block with zeros
        if overlap:
            rxbuffer.write(bytearray(overlap * elementsize))

        # DEBUG
        ## logf = open('chunks2.log', 'wt')
        ## print("delta sleep lag yield misses frames", file=logf)
        ## starttime = dt = rmisses = 0

        wait_time = leadtime = 0
        done = False

        minframes = 1 if varsize else chunksize
        self._offset = offset
        self._frames = frames
        self._pad = pad
        self.start()
        try:
            sleeptime = latency - \
                (rxbuffer.read_available - offset - overlap) / self.samplerate
            if sleeptime > 0:
                _time.sleep(sleeptime)

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
                        ## self._rmisses += 1
                        _time.sleep(0.0025)
                        continue
                elif wait_time:
                    leadtime = lastTime - wait_time
                    wait_time = 0

                # Debugging only
                ## print("{0:f} {1:f} {2:f} {3:f} {4} {5}".format(
                ##     1e3*(_time.time() - starttime), 1e3*sleeptime, 1e3*(self.time - lastTime),
                ##     1e3*dt, self._rmisses - rmisses, frames - chunksize), file=logf)
                ## rmisses = self._rmisses
                ## starttime = _time.time()

                frames, buffregn1, buffregn2 = rxbuffer.get_read_buffers(
                    frames if varsize else chunksize)

                if out is not None or len(buffregn2):
                    buffsz1 = len(buffregn1)
                    bytebuff[:buffsz1] = buffregn1
                    bytebuff[buffsz1:buffsz1 + len(buffregn2)] = buffregn2
                    rxbuffer.advance_read_index(frames - overlap)
                    if not ndarray:
                        yield bytebuff[:frames * elementsize]
                    else:
                        yield tempbuff[:frames]
                else:
                    if atleast_2d:
                        yield numpy.frombuffer(buffregn1, dtype=dtype)\
                                 .reshape(frames, channels)
                    elif ndarray:
                        yield numpy.frombuffer(buffregn1, dtype=dtype)
                    else:
                        yield buffregn1
                    rxbuffer.advance_read_index(frames - overlap)

                # DEBUG
                ## print('1', timer() - gtime, rxbuffer.read_available)
                ## dt = _time.time() - starttime

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

    Other Parameters
    ----------------
    *args, **kwargs
        Arguments to pass to :class:`Stream`.

    """

    def __init__(self, *args, **kwargs):
        super(InputStream, self).__init__('input', *args, **kwargs)

class OutputStream(_OutputStreamMixin, Stream):
    """Playback only stream.

    Other Parameters
    ----------------
    *args, **kwargs
        Arguments to pass to :class:`Stream`.

    """

    def __init__(self, *args, **kwargs):
        super(OutputStream, self).__init__('output', *args, **kwargs)

class DuplexStream(InputStream, OutputStream):
    """Full duplex audio streamer.

    Other Parameters
    ----------------
    *args, **kwargs
        Arguments to pass to :class:`Stream`.

    See Also
    --------
    :class:`OutputStream`, :class:`InputStream`

    """

    def __init__(self, *args, **kwargs):
        Stream.__init__(self, 'duplex', *args, **kwargs)

    @classmethod
    def from_file(cls, playback, *args, **kwargs):
        """Open a stream using the characteristics of a playback Soundfile

        Parameters
        ----------
        playback : SoundFile or str or int or file-like object
            Playback audio file from which to create stream.

        Other Parameters
        ----------------
        *args, **kwargs
            Arguments to pass to Stream constructor

        Returns
        -------
        Stream
            Open stream

        See Also
        --------
        :meth:`InputStream.to_file`

        """
        if not isinstance(playback, _sf.SoundFile):
            playback = _sf.SoundFile(playback)

        if kwargs.get('samplerate', None) is None:
            kwargs['samplerate'] = playback.samplerate

        channels = kwargs.pop('channels', None)
        try:
            kwargs['channels'] = (channels[0], channels[1] or playback.channels)
        except TypeError:
            kwargs['channels'] = (channels, playback.channels)

        return cls(*args, **kwargs)

    def playrec(self, playback, frames=None, pad=0, offset=0, atleast_2d=False,
                loop=False, buffersize=None, blocking=False, out=None):
        """Simultaneously record and play audio data

        Parameters
        -----------
        frames : int, sometimes optional
            Number of frames to play/record. This is required whenever
            `playback` is a file and `out` is not given.
        buffersize : int
            Buffer size to use for (double) buffering audio data to/from
            file. Only applicable when one or both of {`playback`, `out`} is a
            file. Must be a power of 2.
        pad, offset, atleast_2d, loop, blocking, out
            See description of :meth:`InputStream.record` and
            :meth:`OutputStream.play`.

        Returns
        -------
        ndarray or bytearray or type(out)
            Recording destination.

        See Also
        --------
        :meth:`OutputStream.play`, :meth:`InputStream.record`

        """
        try:
            import numpy
        except ImportError:
            numpy = None

        ichannels, ochannels = self.channels
        isamplesize, osamplesize = self.samplesize

        self.set_source(playback, buffersize, loop)

        # if playback is a file with no determined length we could get a
        # nonsensical value which may be negative or a huge number; thus we
        # can't rely on it
        if frames is None and out is None and isinstance(playback, _sf.SoundFile):
            raise TypeError("at least one of {frames, out} is required when playback is a file")
        if out is None:
            if frames is None:
                frames = len(playback)
            if frames < offset:
                raise ValueError("frames must be >= offset")
            out = self._allocate_buffer(frames - offset + (pad if pad >= 0 else 0), 'input')

        if atleast_2d and (numpy is None or not isinstance(out, numpy.ndarray)):
            raise ValueError("atleast_2d is only supported with numpy arrays")

        capture = self.set_sink(out, buffersize)
        isbuffer = not isinstance(out, _sf.SoundFile)
        if frames is not None:
            self._frames = frames
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
           atleast_2d=False, playback=None, loop=False, buffersize=None,
           out=None, **kwargs):
    """Read audio data in iterable chunks from a Portaudio stream.

    Parameters
    ------------
    chunksize, overlap, frames, pad, offset, atleast_2d, playback, loop,
    buffersize, out
        See :meth:`InputStream.chunks` for description.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to Stream constructor.

    Yields
    -------
    ndarray or bytearray or type(out)
        buffer object with `chunksize` elements.

    See Also
    --------
    :meth:`InputStream.chunks`

    """
    if playback is not None:
        if isinstance(playback, _sf.SoundFile):
            stream = DuplexStream.from_file(playback, **kwargs)
        else:
            stream = DuplexStream(**kwargs)
    else:
        stream = InputStream(**kwargs)
    stream._autoclose = True
    return stream.chunks(chunksize, overlap, frames, pad, offset, atleast_2d,
                         playback, loop, buffersize, out)


# Used solely for the pastream app
def _FileStreamFactory(record=None, playback=None, buffersize=None, loop=False, **kwargs):
    frames = kwargs.pop('frames')
    pad = kwargs.pop('pad')
    offset = kwargs.pop('offset')

    iformat, oformat = _sd._split(kwargs.pop('format'))
    isubtype, osubtype = _sd._split(kwargs.pop('subtype'))
    iendian, oendian = _sd._split(kwargs.pop('endian'))

    playback_fh = None
    if playback is not None:
        try:
            playback_fh = _sf.SoundFile(playback)
        except TypeError:
            playback_fh = _sf.SoundFile(playback, kwargs['samplerate'],
                                        ochannels, osubtype, oendian,
                                        oformat)

    if record is not None and playback is not None:
        kind = 'duplex'
        stream = DuplexStream.from_file(playback_fh, **kwargs)
        record_fh = stream.to_file(record, subtype=isubtype, endian=iendian, format=iformat)
        stream.set_source(playback_fh, buffersize, loop)
        stream.set_sink(record_fh, buffersize)
    elif playback is not None:
        kind = 'output'
        record_fh = None
        stream = OutputStream.from_file(playback_fh, **kwargs)
        stream.set_source(playback_fh, buffersize, loop)
    elif record is not None:
        kind = 'input'
        playback_fh = None
        stream = InputStream(**kwargs)
        record_fh = stream.to_file(record, format=iformat, subtype=isubtype, endian=iendian)
        stream.set_sink(record_fh, buffersize)
    else:
        raise TypeError("At least one of {playback, record} must be non-null.")

    # frames/pad/offset can all be specified in seconds (ie a multiple of
    # samplerate) so set them here after the stream is opened
    locs = locals()
    for k in ('frames', 'pad', 'offset'):
        v = locs[k]
        if v is None: continue
        if isinstance(v, str): # parse the format H:M:S to number of seconds
            seconds = sum(float(x)*60**i for i, x in enumerate(reversed(v.split(':'))))
            v = int(round(seconds * stream.samplerate))
        setattr(stream, '_' + k, v)

    origclose = stream.close
    def close():
        try:
            origclose()
        finally:
            if playback is not None:    playback_fh.close()
            if record   is not None:    record_fh.close()
    stream.close = close

    return stream, record_fh, playback_fh, kind


def _get_parser(parser=None):
    import shlex, re
    from argparse import Action, ArgumentParser, RawDescriptionHelpFormatter

    # Hour:minute:second format
    hms = re.compile(r'^\s*([0-9]+:)?([0-5]?[0-9]:)?[0-5]?[0-9](\.[0-9]+)?\s*$')

    if parser is None:
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                                add_help=False, fromfile_prefix_chars='@',
                                description=__doc__)
        parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()

    class ListStreamsAction(Action):
        def __call__(*args, **kwargs):
            print(_sd.query_devices())
            _sys.exit(0)

    def framestype(frames):
        if frames.endswith('s'):
            return sizetype(frames[:-1])
        elif frames == '-1':
            return -1
        elif not hms.match(frames):
            raise ValueError("Couldn't parse argument: %s" % frames)
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

    def nonnegsizetype(x):
        x = sizetype(x)
        assert x >= 0, "Must be a non-negative value."
        return x

    def posframestype(x):
        if x.startswith('-'):
            raise ValueError("Must be a non-negative value.")
        return framestype(x)

    def nullortype(x, type=None):
        if x.lower() in ('null', '', '{}'):
            return None
        return x if type is None else type(x)

    def csvtype(arg, type=None):
        csvsplit = shlex.shlex(arg, posix=True)
        csvsplit.whitespace = ','; csvsplit.whitespace_split = True
        return tuple(map(type or str, csvsplit))

    parser.add_argument("input", type=nullortype, metavar='input|NULL',
        help='''\
Playback audio file. Use dash (-) to read from STDIN. Use one of {null, {}} or
an empty string ('') to select record only.''')

    parser.add_argument("output", type=nullortype, metavar='output|NULL', nargs='?',
        help='''\
Output file for recording. Use dash (-) to write to STDOUT. Use one of {null,
{}} or an empty string ('') to select playback only.''')

    genopts = parser.add_argument_group("general options")

    genopts.add_argument("-h", "--help", action="help",
        help="Show this help message and exit.")

    genopts.add_argument("-l", "--list", action=ListStreamsAction, nargs=0,
        help="List available audio device streams.")

    genopts.add_argument("-q", "--quiet", action='store_true',
        help="Don't print any status information.")

    genopts.add_argument("--version", action='version',
        version='%(prog)s ' + __version__, help="Print version and exit.")

    propts = parser.add_argument_group("playback/record options",
                                       description='''\
Note that size type arguments are accepted in the form hours:minutes:seconds by
default or in samples directly by appending an 's' suffix lead by an optional
size suffix: k[ilo] K[ibi] m[ega] M[ebi]. (e.g. 1Ks == 1024 samples).''')

    propts.add_argument("--buffersize", type=possizetype,
        default=_PA_BUFFERSIZE, help='''\
File buffering size in units of frames. Must be a power of 2. Determines the
maximum amount of buffering for the input/output file(s). Use higher values to
increase robustness against irregular file i/o behavior. (Default
%(default)d)''')

    propts.add_argument("--loop", action='store_true',
        help="Loop playback indefinitely.")

    propts.add_argument("-d", "--duration", type=framestype, default=-1,
        help='''\
Limit playback/capture to a certain duration. If duration is negative (the
default), then streaming will continue until there is no playback data
remaining or, if no playback was given, recording will continue
indefinitely.''')

    propts.add_argument("--fatal-xruns", action='store_true',
        help="Exit with an error if any xruns are reported.")

    propts.add_argument("-o", "--offset", type=posframestype, default=0,
        help='''\
Drop a number of frames from the start of a recording.''')

    propts.add_argument("-p", "--pad", type=framestype, nargs='?', default=0,
        const=-1, help='''\
Pad the input with frames of zeros. (Useful to avoid truncating full duplex
recordings). If pad is negative (the default if no argument is given) then
padding is chosen so that the total playback length matches --duration. If
duration is also negative, zero padding will be added indefinitely.''')

    devopts = parser.add_argument_group("audio device options",
                                        description='''\
Options accept single values or pairs. One of {null, {}) or an empty string
('') may be used as a placeholder for the default value.''')

    devopts.add_argument("-b", "--blocksize", type=nonnegsizetype, help='''\
PortAudio buffer size in units of frames. If blocksize is zero,
backend will decide an optimal size (recommended). Default is zero.''')

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

    fileopts = parser.add_argument_group("audio file formatting options",
                                         description='''\
Options accept single values or pairs. One of {null, {}} or an empty string
('') may be used as a placeholder for the default value.''')

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
    except (TypeError, ValueError):
        traceback.print_exc()
        parser.print_usage()
        parser.exit(255)

    if args.output == '-' or args.quiet:
        stdout = open(os.devnull, 'w')
    else:
        stdout = _sys.stdout

    statline = "\r   {:02.0f}:{:02.0f}:{:02.2f}s ({:d} xruns, {:6.2f}% load)\r"
    print("<-", 'null' if playback is None else playback, file=stdout)
    print("--", stream, file=stdout)
    print("->", 'null' if record is None else record, file=stdout)

    with stream:
        try:
            stream.start()
            t1 = _time.time()
            while stream.active:
                dt = _time.time() - t1
                line = statline.format(dt // 3600, dt % 3600 // 60, dt % 60,
                                       stream.xruns,
                                       100 * stream.cpu_load)
                stdout.write(line); stdout.flush()
                if args.fatal_xruns and stream.status:
                    # I've seen some really odd hanging behavior in older
                    # versions of portaudio using pulseaudio devices with
                    # abort(); stop() seems not to cause the same issues though
                    stream.stop()
                    break
                _time.sleep(0.12)
        except KeyboardInterrupt:
            stream.stop()
        finally:
            print(file=stdout)

        if args.fatal_xruns and stream.xruns:
            print("ERROR: xruns detected: Aborted", file=_sys.stderr)

    print("Callback info:", file=stdout)
    print("\tFrames processed: %d ( %.3fs )"
          % (stream.frame_count, stream.frame_count / float(stream.samplerate)),
          file=stdout)
    print('''\
\tinput xruns (under/over): {0.inputUnderflows}/{0.inputOverflows}
\toutput runs (under/over): {0.outputUnderflows}/{0.outputOverflows}'''
          .format(stream._cstream), file=stdout)

    ## print(stream._rmisses, stream._wmisses, file=_sys.stderr)

    return 0


if __name__ == '__main__':
    _sys.exit(_main())
