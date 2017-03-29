#!/usr/bin/env python
"""
Uses the `soundfile <http://pysoundfile.readthedocs.io>`_ and `sounddevice
<http://python-sounddevice.readthedocs.io>`_ libraries to playback, record, or
simultaneously playback and record audio files.

Notes::

  + 24-bit streaming is currently not supported (typically 32-bit streaming gets
    downconverted automatically anyway)

  + For simplicity, this app only supports 'symmetric' full duplex audio streams;
    i.e., the input device and output device are assumed to be the same.

"""
from __future__ import print_function as _print_function
try:
    import Queue as _queue
except ImportError:
    import queue as _queue
import threading as _threading
import time as _time
import sys as _sys
import sounddevice as _sd
import soundfile as _sf
import weakref as _weakref
from _py_pastream import ffi as _ffi, lib as _lib
try:
    import numpy as _np
except:
    _np = None


__version__ = '0.0.0'
__usage__ = "%(prog)s [options] [-d device] input output"


_PA_BUFFERSIZE = 1<<16 # Default number of frames to buffer i/o to portaudio callback

# Private states to determine how a stream completed
_FINISHED = 1
_ABORTED = 2
_STOPPED = 4

paInputOverflow = _lib.paInputOverflow
paInputUnderflow = _lib.paInputUnderflow
paOutputOverflow = _lib.paOutputOverflow
paOutputUnderflow = _lib.paOutputUnderflow

class PaStreamException(Exception):
    pass
 
class AudioBufferError(PaStreamException):
    pass

class XRunError(AudioBufferError):
    pass

class ReceiveBufferFull(AudioBufferError):
    pass

class TransmitBufferEmpty(AudioBufferError):
    pass

class RingBuffer(object):
    # Copyright (c) 2017 Matthias Geier
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.
    """Wrapper for PortAudio's ring buffer.

    See __init__().

    """

    def __init__(self, elementsize, size):
        """Create an instance of PortAudio's ring buffer.

        Parameters
        ----------
        elementsize : int
            The size of a single data element in bytes.
        size : int
            The number of elements in the buffer (must be a power of 2).

        """
        self._ptr = _ffi.new('PaUtilRingBuffer*')
        self._data = _ffi.new('unsigned char[]', size * elementsize)
        res = _lib.PaUtil_InitializeRingBuffer(
            self._ptr, elementsize, size, self._data)
        if res != 0:
            assert res == -1
            raise ValueError('size must be a power of 2')
        assert self._ptr.bufferSize == size
        assert self._ptr.elementSizeBytes == elementsize

    def flush(self):
        """Reset buffer to empty.

        Should only be called when buffer is NOT being read or written.

        """
        _lib.PaUtil_FlushRingBuffer(self._ptr)

    @property
    def write_available(self):
        """Number of elements available in the ring buffer for writing."""
        return _lib.PaUtil_GetRingBufferWriteAvailable(self._ptr)

    @property
    def read_available(self):
        """Number of elements available in the ring buffer for reading."""
        return _lib.PaUtil_GetRingBufferReadAvailable(self._ptr)

    def write(self, data, size=-1):
        """Write data to the ring buffer.

        Parameters
        ----------
        data : CData pointer or buffer or bytes
            Data to write to the buffer.
        size : int, optional
            The number of elements to be written.

        Returns
        -------
        int
            The number of elements written.

        """
        try:
            data = _ffi.from_buffer(data)
        except TypeError:
            pass  # input is not a buffer
        if size < 0:
            size, rest = divmod(_ffi.sizeof(data), self._ptr.elementSizeBytes)
            if rest:
                raise ValueError('data size must be multiple of elementsize')
        return _lib.PaUtil_WriteRingBuffer(self._ptr, data, size)

    def read(self, data, size=-1):
        """Read data from the ring buffer.

        Parameters
        ----------
        data : CData pointer or buffer
            The memory where the data should be stored.
        size : int, optional
            The number of elements to be read.

        Returns
        -------
        int
            The number of elements read.

        """
        try:
            data = _ffi.from_buffer(data)
        except TypeError:
            pass  # input is not a buffer
        if size < 0:
            size, rest = divmod(_ffi.sizeof(data), self._ptr.elementSizeBytes)
            if rest:
                raise ValueError('data size must be multiple of elementsize')
        return _lib.PaUtil_ReadRingBuffer(self._ptr, data, size)

    def get_write_buffers(self, size):
        """Get buffer(s) to which we can write data.

        Parameters
        ----------
        size : int
            The number of elements desired.

        Returns
        -------
        int
            The room available to be written or the given *size*,
            whichever is smaller.
        buffer
            The first buffer.
        buffer
            The second buffer.

        """
        ptr1 = _ffi.new('void**')
        ptr2 = _ffi.new('void**')
        size1 = _ffi.new('ring_buffer_size_t*')
        size2 = _ffi.new('ring_buffer_size_t*')
        return (_lib.PaUtil_GetRingBufferWriteRegions(
                    self._ptr, size, ptr1, size1, ptr2, size2),
                _ffi.buffer(ptr1[0], size1[0] * self.elementsize),
                _ffi.buffer(ptr2[0], size2[0] * self.elementsize))

    def advance_write_index(self, size):
        """Advance the write index to the next location to be written.

        Parameters
        ----------
        size : int
            The number of elements to advance.

        Returns
        -------
        int
            The new position.

        """
        return _lib.PaUtil_AdvanceRingBufferWriteIndex(self._ptr, size)

    def get_read_buffers(self, size):
        """Get buffer(s) from which we can read data.

        Parameters
        ----------
        size : int
            The number of elements desired.

        Returns
        -------
        int
            The number of elements available for reading.
        buffer
            The first buffer.
        buffer
            The second buffer.

        """
        ptr1 = _ffi.new('void**')
        ptr2 = _ffi.new('void**')
        size1 = _ffi.new('ring_buffer_size_t*')
        size2 = _ffi.new('ring_buffer_size_t*')
        return (_lib.PaUtil_GetRingBufferReadRegions(
                    self._ptr, size, ptr1, size1, ptr2, size2),
                _ffi.buffer(ptr1[0], size1[0] * self.elementsize),
                _ffi.buffer(ptr2[0], size2[0] * self.elementsize))

    def advance_read_index(self, size):
        """Advance the read index to the next location to be read.

        Parameters
        ----------
        size : int
            The number of elements to advance.

        Returns
        -------
        int
            The new position.

        """
        return _lib.PaUtil_AdvanceRingBufferReadIndex(self._ptr, size)

    @property
    def elementsize(self):
        """Element size in bytes."""
        return self._ptr.elementSizeBytes

    def __len__(self):
        """Size of buffer in elements"""
        return self._ptr.bufferSize

class _BufferedStreamBase(_sd._StreamBase):
    """
    This class adds a RingBuffer for reading and writing audio
    data. This double buffers the audio data so that any processing is
    kept out of the time sensitive audio callback function.

    Notes:

    If the receive buffer fills during a callback the audio stream is
    aborted and an exception is raised.

    During playback, the end of the stream is signaled by an item on
    the queue that is smaller than blocksize*channels*samplesize
    bytes.

    This class adds the ability to register functions (`reader`,
    `writer`) for reading and writing audio data which run in their own
    threads. However, the reader and writer threads are optional; this
    allows the use of a 'duplex' stream which e.g. has a dedicated
    thread for writing data but for which receive data is read directly
    in the owning thread.

    Parameters
    -----------
    nframes : int
        Number of frames to play/record. (0 means unlimited). This does *not*
        include the length of any additional padding.
    padding : int
        Number of zero frames to pad the output with. This has no effect on the
        input.
    offset : int
        Number of frames to discard from beginning of input. This has no effect
        on the output.
    buffersize : int
        Transmit/receive buffer size in units of frames. Increase for smaller
        blocksizes.
    reader, writer : function
        Buffer reader and writer functions to be run in a separate
        thread.
    raise_on_xruns : bool or int
        Abort the stream on an xrun condition and raise an XRunError
        exception. If True the stream will be aborted on any xrun
        condition. Alternatively, pass a combination of
        pa{Input,Output}{Overflow,Underflow} to only raise on certain
        xrun conditions.
    blocksize : int or None
        Portaudio buffer size. If None or 0, the Portaudio backend will
        automatically determine a size.

    Other Parameters
    ----------------
    kind, device, channels, dtype, **kwargs
        Additional parameters to pass to `sounddevice._StreamBase`.

    Attributes
    ----------
    txbuff : Ringbuffer
        RingBuffer used for writing audio data to the output Portaudio stream.
    rxbuff : Ringbuffer
        RingBuffer used for reading audio data from the input Portaudio stream.
    framesize : int
        The audio frame size in bytes. Equivalent to channels*samplesize.
    """
    def __init__(self, kind, nframes=0, padding=0, offset=0,
                 buffersize=_PA_BUFFERSIZE, reader=None, writer=None,
                 raise_on_xruns=False, blocksize=None, device=None,
                 channels=None, dtype=None, **kwargs):
        # unfortunately we need to figure out the framesize before allocating
        # the stream in order to be able to pass our user_data
        self.txbuff = self.rxbuff = None
        if kind == 'duplex':
            idevice, odevice = _sd._split(device)
            ichannels, ochannels = _sd._split(channels)
            idtype, odtype = _sd._split(dtype)
            iparameters, idtype, isize, _ = _sd._get_stream_parameters(
                'input', idevice, ichannels, idtype, None, None, None)
            oparameters, odtype, osize, _ = _sd._get_stream_parameters(
                'output', odevice, ochannels, odtype, None, None, None)

            self.framesize = (isize*iparameters.channelCount, 
                              osize*oparameters.channelCount)
            self.txbuff = RingBuffer(self.framesize[1], buffersize)
            self.rxbuff = RingBuffer(self.framesize[0], buffersize)
        else:
            parameters, dtype, samplesize, _ = _sd._get_stream_parameters(
                kind, device, channels, dtype, None, None, None)
            self.framesize = samplesize*parameters.channelCount
            if kind == 'output':
                self.txbuff = RingBuffer(self.framesize, buffersize)
            if kind == 'input':
                self.rxbuff = RingBuffer(self.framesize, buffersize)

        # Set up the C portaudio callback
        self._cstream = _ffi.NULL
        self.__weakref = _weakref.WeakKeyDictionary()
        self.__nframes = nframes
        if kwargs.get('callback', None) is None:
            cbinfo = _ffi.new("Py_PaCallbackInfo*")
            userdata = {'duplexity': ['input', 'output', 'duplex'].index(kind) + 1,
                        'abort_on_xrun': 0xF if raise_on_xruns is True else int(raise_on_xruns),
                        'padding': padding,
                        'callbackInfo': cbinfo,
                        'offset': offset }
            if self.rxbuff is not None:
                userdata['rxbuff'] = _ffi.cast('PaUtilRingBuffer*', self.rxbuff._ptr)
            if self.txbuff is not None:
                userdata['txbuff'] = _ffi.cast('PaUtilRingBuffer*', self.txbuff._ptr)
                
            self._cstream = _ffi.new("Py_PaBufferedStream*", userdata)
            self.__weakref[self._cstream] = cbinfo
            kwargs['userdata'] = self._cstream
            kwargs['callback'] = _ffi.addressof(_lib, 'callback')
            kwargs['wrap_callback'] = None
        self._reset_stream_info()

        # These flags are used to tell when the callbacks have
        # finished.  We can use them to abort writing of the
        # ringbuffer.
        self.__statecond = _threading.Condition()
        self.__streamlock = _threading.RLock()
        self.__state = 0
        self.__aborting = False
        self.__exceptions = _queue.Queue()

        # set up reader/writer threads
        self._owner_thread = _threading.current_thread()
        self._txthread = None
        if (kind == 'duplex' or kind == 'output') and writer is not None:
            self._txthread_args = {'target': self._readwritewrapper,
                                   'args': (self.txbuff, writer)}
        else:
            self._txthread_args = None
        self._rxthread = None
        if (kind == 'duplex' or kind == 'input') and reader is not None:
            self._rxthread_args = {'target': self._readwritewrapper,
                                   'args': (self.rxbuff, reader)}
        else:
            self._rxthread_args = None

        # TODO: add support for C finished_callback function pointer
        user_callback = kwargs.get('finished_callback', lambda : None)
        def finished_callback():
            # Check for any errors that might've occurred in the
            # callback
            # Note: exceptions from the callback take precedence over
            # any xrun exceptions
            msg = _ffi.string(self._cstream.errorMsg).decode('utf-8')
            if len(msg):
                if ':' in msg:
                    exctype, excmsg = msg.split(':', 1)
                else:
                    exctype, excmsg = msg, ''
                exctype = getattr(_sys.modules[__name__], exctype)
                if exctype is XRunError and not len(excmsg):
                    self._set_exception(XRunError(str(self.status)))
                else:
                    self._set_exception(exctype(excmsg))
                
            with self.__statecond:
                # It's possible that the callback aborted itself so check
                # if we need to update our aborted flag here
                if (self._cstream.last_callback == _sd._lib.paAbort
                    or self.__aborting
                    or not self.__exceptions.empty()):
                    self.__state = _ABORTED | _FINISHED
                elif self._cstream.last_callback == _sd._lib.paComplete:
                    self.__state = _FINISHED
                else:
                    self.__state = _STOPPED | _FINISHED
                self.__statecond.notify_all()

            try:
                # Okay, let the user handle their stuff
                user_callback()                
            except:
                self._set_exception()

        super(_BufferedStreamBase, self).__init__(kind, blocksize=blocksize,
                                                  device=device,
                                                  channels=channels,
                                                  dtype=dtype,
                                                  finished_callback=finished_callback,
                                                  **kwargs)

        self._rmisses = self._wmisses = 0
        if kind == 'duplex':
            self._device_name = tuple(_sd.query_devices(dev)['name'] for dev in self._device)
        else:
            self._device_name = _sd.query_devices(self._device)['name']

    def __stopiothreads(self):
        currthread = _threading.current_thread()
        if (self._rxthread is not None
            and self._rxthread.is_alive()
            and self._rxthread != currthread):
            self._rxthread.join()
        if (self._txthread is not None
            and self._txthread.is_alive()
            and self._txthread != currthread):
            self._txthread.join()

    def _readwritewrapper(self, buff, rwfunc):
        """\
        Wrapper for the reader and writer functions which acts as a kind
        of context manager.
        """
        try:
            rwfunc(self, buff)
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

        # To simplify things, we only care about the first exception
        # raised
        # TODO: handle multiple deferred exceptions
        self.__exceptions.queue.clear()

        if isinstance(exc, tuple):
            exctype, excval, exctb = exc
            if exctype is not None:
                excval = exctype(excval)
            try:
                raise excval.with_traceback(exctb)
            except AttributeError:
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
        Block until portaudio stream has finished processing all data or
        has been aborted/stopped.

        NB: This function does not wait for receive buffer to empty.
        """
        with self.__statecond:
            if self.__state == 0:
                self.__statecond.wait(timeout)
                self._reraise_exceptions()
            return self.__state > 0

    @property
    def aborted(self):
        """\
        Whether stream has been aborted. If True, it is
        guaranteed that the stream is in a finished state.
        """
        return self.__state & _ABORTED > 0
    
    @property
    def finished(self):
        """\
        Whether the portaudio stream is in a finished state. Will only
        be True if `start()` has been called and the stream either
        completed sucessfully or was stopped/aborted.
        """
        return self.__state & _FINISHED > 0

    @property
    def status(self):
        """\
        The current PaStreamCallbackFlags status of the portaudio
        stream.
        """
        return _sd.CallbackFlags(self._cstream.status)

    @property
    def _callback_info(self):
        return self._cstream.callbackInfo

    @property
    def xruns(self):
        return self._cstream.xruns

    @property
    def frame_count(self):
        """\
        Running total of frames that have been processed. Each new
        starting of the stream resets this number to zero.
        """
        return self._cstream.frame_count

    @property
    def padding(self):
        """\
        Number of frames of padding that will be added to the input
        stream.
        """
        return self._cstream.padding

    @padding.setter
    def padding(self, value):
        # Note that the py_pastream callback doesn't act on `padding` unless
        # nframes == 0; thus, set `nframes` first to get deterministic
        # behavior.
        if self.__nframes:
            self.__nframes = self._cstream.nframes = value + self.__nframes
        self._cstream.padding = value
    
    @property
    def nframes(self):
        """\
        Total number of frames to be processed by the stream. Returns
        zero if no limit is set.
        """
        # We fib a bit here: __nframes != _cstream.nframes. The former doesn't
        # include any padding.
        return self.__nframes

    @nframes.setter
    def nframes(self, value):
        # Set nframes in an atomic manner
        if value:
            self._cstream.nframes = value + self._cstream.padding
        else:
            self._cstream.nframes = value
        self.__nframes = value

    def __enter__(self):
        return self

    def __exit__(self, exctype, excvalue, exctb):
        self.close()

    def _reset_stream_info(self):
        self._cstream.lastTime = 0
        self._cstream.last_callback = _sd._lib.paContinue
        self._cstream.status = 0
        self._cstream.frame_count = 0
        self._cstream.errorMsg = b''
        self._cstream.xruns = 0
        self._cstream.inputUnderflows = 0
        self._cstream.inputOverflows = 0
        self._cstream.outputUnderflows = 0
        self._cstream.outputOverflows = 0
        self._cstream._nframesIsUnset = 0
        self.nframes = self.__nframes

        for i in range(_lib.MEASURE_LEN):
            self._cstream.callbackInfo.period[i] = 0
        self._cstream.callbackInfo.call_count = 0
        self._cstream.callbackInfo.min_dt = -1
        self._cstream.callbackInfo.max_dt = -1

    def _prepare(self):
        assert not self.active, "Stream has already been started!"

        # Apparently when using a PaStreamFinishedCallback the stream
        # *must* be stopped before starting the stream again or the
        # streamFinishedCallback will never be called
        super(_BufferedStreamBase, self).stop()

        # Reset stream state
        with self.__statecond:
            self.__state = 0
        with self.__streamlock:
            self.__aborting = False
        self._reset_stream_info()
        if self.rxbuff is not None:
            self.rxbuff.flush()

        # Recreate the necessary threads
        if self._rxthread_args is not None:
            self._rxthread = _threading.Thread(**self._rxthread_args)
            self._rxthread.daemon = True
        if self._txthread_args is not None:
            self._txthread = _threading.Thread(**self._txthread_args)
            self._txthread.daemon = True

    def start(self):
        self._prepare()
        if self._txthread is not None:
            self._txthread.start()
            while self.txbuff.write_available and self._txthread.is_alive():
                _time.sleep(0.05)
            self._reraise_exceptions()
        super(_BufferedStreamBase, self).start()
        if self._rxthread is not None:
            self._rxthread.start()
            self._reraise_exceptions()

    def stop(self):
        with self.__streamlock:
            super(_BufferedStreamBase, self).stop()
        self.__stopiothreads()
        self._reraise_exceptions()

    def abort(self):
        with self.__streamlock:
            self.__aborting = True
            super(_BufferedStreamBase, self).abort()
        self.__stopiothreads()
        self._reraise_exceptions()

    def close(self):
        # we take special care here to abort the stream first so that it
        # is still valid for the lifetime of the read/write threads
        with self.__streamlock:
            self.__aborting = True
            super(_BufferedStreamBase, self).abort()
        self.__stopiothreads()
        with self.__streamlock:
            super(_BufferedStreamBase, self).close()
        self._reraise_exceptions()

    def __repr__(self):
        if not isinstance(self.device, int) and self.device[0] == self.device[1]:
            device_name = self._device_name[0]
        else:
            device_name = self._device_name
        if not isinstance(self.channels, int) and self.channels[0] == self.channels[1]:
            channels = self.channels[0]
        else:
            channels = self.channels
        if self.dtype[0] == self.dtype[1]:
            dtype = self.dtype[0]
        else:
            dtype = self.dtype

        return ("{0}({1!r}, samplerate={4._samplerate:.0f}, "
                "channels={2}, dtype={3!r}, blocksize={4._blocksize})"
                ).format(self.__class__.__name__, device_name, channels, dtype, self)

class _InputStreamMixin(object):
             
    # TODO: add buffer 'type' as an argument
    # TODO: add fill_value option
    def chunks(self, chunksize=0, overlap=0, always_2d=False, copy=False):
        """
        Similar to SoundFile.blocks. Returns an iterator over buffered audio
        chunks read from a Portaudio stream.

        Parameters
        ----------
        chunksize : int
            Size of chunks. This is aside from the stream `blocksize`
            and is required if `blocksize` is zero or unset.
        overlap : int
            Number of frames to overlap across blocks.
        always_2d : bool
            Always returns blocks 2 dimensional arrays. Only valid if you have
            numpy installed.
        copy : bool
            Whether to return copies of blocks. By default a view is returned.

        Yields
        ------
        array
            ndarray or memoryview object with `chunksize` or `blocksize` elements.
        """
        chunksize = chunksize or self.blocksize
        if not chunksize:
            raise ValueError("One of block or chunk size must be set")
        if overlap and not chunksize:
            raise ValueError("Using overlap requires a fixed known chunk or block size")
        if overlap >= chunksize:
            raise ValueError("Overlap must be less than chunk or block isze")
        if _np is None and always_2d:
            raise ValueError("always_2d is only supported with numpy")

        try:
            channels = self.channels[0]
            dtype = self.dtype[0]
            framesize = self.framesize[0]
        except TypeError:
            dtype = self.dtype
            channels = self.channels
            framesize = self.framesize

        if channels > 1:
            always_2d = True

        copy |= overlap
        if _np is None:
            tempbuff = memoryview(bytearray(chunksize*framesize))
            copier = lambda x: memoryview(bytearray(buff))
        else:
            tempbuff = _np.zeros((chunksize, channels) if always_2d else chunksize*channels, dtype=dtype)
            copier = _np.copy

        incframes = chunksize - overlap
        rxbuff = self.rxbuff

        runtime = 0
        done = False
        self.start()
        sleeptime = (incframes - rxbuff.read_available)  / self.samplerate
        if sleeptime > 0:
            _time.sleep(sleeptime)
        while not (self.aborted or done):
            # for thread safety, check the stream is active *before* reading
            active = self.active 
            nframes = min(rxbuff.read_available, incframes)
            if nframes < incframes:
                if not active:
                    done = True
                else:
                    self._rmisses += 1 if self._callback_info.call_count > 0 else 0
                    _time.sleep(0.0005)
                    continue

            runtime = _time.time()
            nframes, buffregn1, buffregn2 = rxbuff.get_read_buffers(nframes)
            if len(buffregn2) or copy:
                # FIXME: this only works for ndarrays atm
                tempbuff[:len(buffregn1)//rxbuff.elementsize].data = buffregn1
                tempbuff[:len(buffregn2)//rxbuff.elementsize].data = buffregn2
                rxbuff.advance_read_index(nframes)
                yield copier(tempbuff[:overlap + nframes]) if copy else tempbuff[:overlap + nframes]
            elif _np:
                yield _np.frombuffer(buffregn1, dtype=dtype).reshape(nframes, channels) 
                rxbuff.advance_read_index(nframes)
            else:
                yield buffregn1
                rxbuff.advance_read_index(nframes)
            if overlap:
                tempbuff[:-incframes] = tempbuff[incframes:]

            sleeptime = incframes / self.samplerate - (_time.time() - runtime)
            if sleeptime > 0:
                _time.sleep(sleeptime)

class BufferedInputStream(_InputStreamMixin, _BufferedStreamBase):
    def __init__(self, **kwargs):
        super(BufferedInputStream, self).__init__('input', **kwargs)

class BufferedOutputStream(_BufferedStreamBase):
    def __init__(self, **kwargs):
        super(BufferedOutputStream, self).__init__('output', **kwargs)

class BufferedStream(BufferedInputStream, BufferedOutputStream):
    def __init__(self, **kwargs):
        _BufferedStreamBase.__init__(self, 'duplex', **kwargs)

class _SoundFileStreamBase(_BufferedStreamBase):
    """
    This helper class basically gives you two things:

        1) it provides complete reader and writer functions for SoundFile
           objects (or anything that can be opened as a SoundFile object)

        2) it automatically sets parameters for the stream based on the input
           file and automatically sets parameters for the output file based on
           the output stream.

    Parameters
    ----------
    inpf : SoundFile compatible input
        Input file to stream to audio device. The input file will determine the
        samplerate and number of channels for the audio stream.
    outf : SoundFile compatible input
        Output file to capture data from audio device. If a SoundFile is not
        passed, the output file parameters will be determined from the output
        audio stream.

    Attributes
    ------------
    inp_fh : SoundFile
        The file object to write to the output ring buffer.
    out_fh : SoundFile
        The file object to capture data from the input ring buffer.

    Other Parameters
    ----------------------
    reader, writer, kind, blocksize, **kwargs
        Additional parameters to pass to _BufferedStreamBase.
    """
    def __init__(self, kind, inpf=None, outf=None, reader=None,
                 writer=None, sfkwargs={}, blocksize=None, **kwargs):
        # At this point we don't care what 'kind' the stream is, only whether
        # the input/output is None which determines whether reader/writer
        # functions should be registered
        self._inpf = inpf
        if self._inpf is not None:
            if not isinstance(self._inpf, _sf.SoundFile):
                self.inp_fh = _sf.SoundFile(self._inpf)
            else:
                self.inp_fh = self._inpf
            if kwargs.get('samplerate', None) is None:
                kwargs['samplerate'] = self.inp_fh.samplerate
            if kwargs.get('channels', None) is None:
                kwargs['channels'] = self.inp_fh.channels
            if writer is None:
                writer = self._soundfilereader
        else:
            self.inp_fh = self._inpf

        # We need to set the reader here; output file parameters will known
        # once we open the stream
        self._outf = outf
        if outf is not None and reader is None:
            reader = self._soundfilewriter
        self.out_fh = None

        super(_SoundFileStreamBase, self).__init__(kind, reader=reader,
                                                   writer=writer,
                                                   blocksize=blocksize,
                                                   **kwargs)

        # Try and determine the file extension here; we need to know it if we
        # want to try and set a default subtype for the output
        try:
            outext = getattr(outf, 'name', outf).rsplit('.', 1)[1].lower()
        except (AttributeError, IndexError):
            outext = None

        # If the output file hasn't already been opened, we open it here using
        # the input file and output stream settings, plus any user supplied
        # arguments
        if not (self._outf is None or isinstance(self._outf, _sf.SoundFile)):
            if self.inp_fh is not None:
                if sfkwargs.get('endian', None) is None:
                    sfkwargs['endian'] = self.inp_fh.endian
                if (sfkwargs.get('subtype', None) is None
                    and _sf.check_format(sfkwargs.get('format', None) or outext, self.inp_fh.subtype)):
                    sfkwargs['subtype'] = self.inp_fh.subtype
            if sfkwargs.get('channels', None) is None:
                sfkwargs['channels'] = self.channels[0] if kind == 'duplex' else self.channels
            if sfkwargs.get('samplerate', None) is None:
                sfkwargs['samplerate'] = int(self.samplerate)
            if sfkwargs.get('mode', None) is None:
                sfkwargs['mode'] = 'w+b'
            self.out_fh = _sf.SoundFile(self._outf, **sfkwargs)
        else:
            self.out_fh = self._outf

    # Default handler for writing input from a Buffered to a SoundFile object
    @staticmethod
    def _soundfilewriter(stream, rxbuff):
        try:               
            framesize = stream.framesize[0]
            dtype = stream.dtype[0]
        except TypeError: 
            framesize = stream.framesize
            dtype = stream.dtype    

        dt = len(rxbuff) / stream.samplerate / 8
        while not stream.aborted:
            # for thread safety, check the stream is active *before* reading
            active = stream.active 
            nframes = rxbuff.read_available
            if nframes == 0:
                # we've read everything and the stream is done; seeya!
                if not active: break
                # we're reading too fast, wait for a buffer write
                stream._rmisses += 1
                stream.wait(dt)
                continue

            nframes, buffregn1, buffregn2 = rxbuff.get_read_buffers(nframes)
            stream.out_fh.buffer_write(buffregn1, dtype=dtype)
            if len(buffregn2):
                stream.out_fh.buffer_write(buffregn2, dtype=dtype)
            rxbuff.advance_read_index(nframes)

    # Default handler for reading input from a SoundFile object and
    # writing it to a BufferedStream
    @staticmethod
    def _soundfilereader(stream, txbuff):
        try:               
            framesize = stream.framesize[1]
            dtype = stream.dtype[1]
        except TypeError: 
            framesize = stream.framesize
            dtype = stream.dtype    

        dt = len(txbuff) / stream.samplerate / 8
        while not stream.finished:
            nframes = txbuff.write_available
            if not nframes:
                stream.wait(dt)
                continue

            nframes, buffregn1, buffregn2 = txbuff.get_write_buffers(nframes)
            readframes = stream.inp_fh.buffer_read_into(buffregn1, dtype=dtype)
            if len(buffregn2):
                readframes += stream.inp_fh.buffer_read_into(buffregn2, dtype=dtype)
            txbuff.advance_write_index(readframes)

            if readframes < nframes:
                break # we've reached end of file; all done!

    def close(self):
        try:
            super(_SoundFileStreamBase, self).close()
        finally:
            if not (self._outf is None or isinstance(self._outf, _sf.SoundFile)):
                self.out_fh.close()
            if not (self._inpf is None or isinstance(self._inpf, _sf.SoundFile)):
                self.inp_fh.close()

class SoundFileInputStream(_InputStreamMixin, _SoundFileStreamBase):
    """
    Audio file recorder.

    Parameters
    -----------
    outf : SoundFile compatible input
        Output file to capture data from audio device. If a SoundFile is not
        passed, the output file parameters will be determined from the output
        audio stream.
    sfkwargs : dict
        Arguments to pass when creating SoundFile when outf is not already a
        SoundFile object. This allows overriding of any of the default
        parameters.

    Other Parameters
    -----------------
    **kwargs
        Additional parameters to pass to SoundFileStreamBase.
    """
    def __init__(self, outf, sfkwargs={}, **kwargs):
        super(SoundFileInputStream, self).__init__('input', outf=outf,
                                                   sfkwargs=sfkwargs,
                                                   **kwargs)

class SoundFileOutputStream(_SoundFileStreamBase):
    """
    Audio file player.

    Parameters
    -----------
    inpf : SoundFile compatible input
        Input file to stream to audio device. The input file will determine the
        samplerate and number of channels for the audio stream.
    """
    def __init__(self, inpf, buffersize=_PA_BUFFERSIZE, **kwargs):
        super(SoundFileOutputStream, self).__init__('output', inpf=inpf,
                                                    buffersize=buffersize,
                                                    **kwargs)

class SoundFileStream(SoundFileInputStream, SoundFileOutputStream):
    """
    Full duplex audio file streamer. Note that only one of inpf and outf
    is required. This allows you to e.g. use a SoundFile as input but
    implement your own reader and/or read from the buffer in the
    stream's owner thread.
    """
    def __init__(self, inpf=None, outf=None, buffersize=_PA_BUFFERSIZE,
                 sfkwargs={}, **kwargs):
        # If you're not using soundfiles for the input or the output,
        # then you should probably be using the BufferedStream class
        if inpf is None and outf is None:
            raise ValueError("No input or output file given.")

        _SoundFileStreamBase.__init__(self, 'duplex', inpf=inpf,
                                      outf=outf, buffersize=buffersize,
                                      sfkwargs=sfkwargs, **kwargs)

def chunks(inpf=None, chunksize=0, overlap=0, always_2d=False, copy=False,
           writer=None, streamclass=None, **kwargs):
    """
    Read audio data in chunks from a Portaudio stream. Can be either half-duplex
    (recording-only) or full-duplex (if an input file is supplied).

    Parameters
    ------------
    inpf : SoundFile compatible input or None, optional
        Optional input stimuli.
    chunksize : int
        Size of chunks. This is aside from the stream `blocksize` and is
        required if `blocksize` is zero or unset.
    overlap : int, optinal
        Number of frames to overlap across blocks.
    always_2d : bool, optional
        Always returns blocks 2 dimensional arrays. Only valid if you have
        numpy installed.
    copy : bool, optional
        Whether to return copies of blocks. By default a view is returned.

    Other Parameters
    -----------------
    writer : function
        Function that handles writing to the audio transmit buffer. Can be used
        as an alternative to or in combination with an input file.
    streamclass : object
        Base class to use. Typically this is automatically determined from the input arguments.
    **kwargs
        Additional arguments to pass to base stream class.

    Yields
    -------
    array
        ndarray or memoryview object with `blocksize` elements.
    """
    if inpf is None:
        if streamclass is None:
            streamclass = BufferedInputStream
        stream = streamclass(**kwargs)
    else:
        if streamclass is None:
            streamclass = SoundFileStream
        stream = streamclass(inpf, **kwargs)

    with stream:
        for blk in stream.blocks(chunksize, overlap, always_2d, copy):
            yield blk 

# Used just for the pastream app
def _SoundFileStreamFactory(inpf=None, outf=None, **kwargs):
    if inpf is not None and outf is not None:
        Streamer = SoundFileStream
        ioargs = (inpf, outf)
        kind = 'duplex'
    elif inpf is not None:
        Streamer = SoundFileOutputStream
        ioargs = (inpf,)
        kwargs.pop('sfkwargs', None)
        kwargs.pop('reader', None)
        kind = 'input'
    elif outf is not None:
        Streamer = SoundFileInputStream
        ioargs = (outf,)
        kwargs.pop('writer', None)
        kind = 'output'
    else:
        raise SystemExit("No input or output selected.")

    return Streamer(*ioargs, **kwargs), kind

def _get_parser(parser=None):
    from argparse import Action, ArgumentParser, RawDescriptionHelpFormatter

    if parser is None:
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                                fromfile_prefix_chars='@',
                                usage=__usage__,
                                description=__doc__)
        parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()

    class ListStreamsAction(Action):
        def __call__(*args, **kwargs):
            print(_sd.query_devices())
            _sys.exit(0)

    def devtype(dev):
        try:               return int(dev)
        except ValueError: return dev

    def posint(intarg):
        intarg = int(intarg)
        assert intarg > 0, "Must be a positive value."
        return intarg

    parser.add_argument("input", type=lambda x: None if x=='-' else x,
                        help='''\
Input audio file, or, use the special designator '-' for recording only.''')

    parser.add_argument("output", type=lambda x: None if x=='-' else x,
                        help='''\
Output recording file, or, use the special designator '-' for playback only.''')

#     parser.add_argument("--loop", default=False, nargs='?', metavar='n', const=True, type=int,
#                         help='''\
# Replay the playback file n times. If no argument is specified, playback will
# loop infinitely. Does nothing if there is no playback.''')

    parser.add_argument("-l", action=ListStreamsAction, nargs=0,
                       help="List available audio device streams.")

    parser.add_argument("-q", "--qsize", type=posint, default=_PA_BUFFERSIZE,
help="File transmit buffer size (in units of frames). Increase for smaller blocksizes.")

    parser.add_argument("-p", "--pad", type=posint, default=0,
                        help='''\
Pad the input with frames of zeros. Useful to avoid truncating a full duplex
recording.''')

    parser.add_argument("-o", "--offset", type=posint, default=0,
                        help='''\
Drop a number of frames from the start of a recording.''')

    parser.add_argument("-n", "--nframes", type=posint, default=0,
                        help='''\
Limit playback/capture to this many frames.''')

    devopts = parser.add_argument_group("I/O device stream options")

    devopts.add_argument("-d", "--device", type=devtype,
                         help='''\
Audio device name expression or index number. Defaults to the PortAudio default device.''')

    devopts.add_argument("-b", "--blocksize", type=int,
                         help='''\
PortAudio buffer size in units of frames. If zero or not specified,
backend will decide an optimal size.''')

    devopts.add_argument("-f", "--format", dest='dtype',
                         choices=_sd._sampleformats.keys(),
                         help='''Sample format of device I/O stream.''')

    devopts.add_argument("-c", "--channels", type=int,
                         help="Number of channels.")

    devopts.add_argument("-r", "--rate", dest='samplerate',
                         type=lambda x: int(float(x[:-1])*1000) if x.endswith('k') else int(x),
                         help="Sample rate in Hz. Add a 'k' suffix to specify kHz.")

    fileopts = parser.add_argument_group('''\
Output file format options''')

    fileopts.add_argument("-t", dest="file_type", choices=_sf.available_formats().keys(),
                          type=str.upper,
                          help='''\
Output file type. Typically this is determined from the file extension, but it
can be manually overridden here.''')

    fileopts.add_argument("-e", dest="encoding", choices=_sf.available_subtypes(),
                          type=str.upper,
                          help="Sample format encoding.")

    fileopts.add_argument("--endian", choices=['file', 'big', 'little'],
                          help="Byte endianness.")

    return parser

def _main(argv=None):
    if argv is None: argv=_sys.argv[1:]
    parser = _get_parser()
    args = parser.parse_args(argv)

    sfkwargs=dict()
    stream, kind = _SoundFileStreamFactory(args.input, args.output,
                                           buffersize=args.qsize,
                                           nframes=args.nframes,
                                           padding=args.pad,
                                           offset=args.offset,
                                           sfkwargs={
                                               'endian': args.endian,
                                               'subtype': args.encoding,
                                               'format': args.file_type
                                           },
                                           device=args.device,
                                           channels=args.channels,
                                           dtype=args.dtype,
                                           samplerate=args.samplerate,
                                           blocksize=args.blocksize)

    statline = '''\
\r{:8.3f}s {:10d} frames processed, {:>8s} frames free, {:>8s} frames queued ({:d} xruns, {:f}% load)\r'''
    print("<--", stream.inp_fh if stream.inp_fh is not None else 'null')
    print(["<->", "-->", "<--"][['duplex', 'input', 'output'].index(kind)], stream)
    print("-->", stream.out_fh if stream.out_fh is not None else 'null')

    nullinp = nullout = None
    if stream.inp_fh is None:
        nullinp = 'n/a'
    if stream.out_fh is None:
        nullout = 'n/a'

    try:
        try:
            with stream:
                stream.start()
                t1 = _time.time()
                while stream.active:
                    _time.sleep(0.1)
                    line = statline.format(_time.time()-t1, stream.frame_count,
                                           nullinp or str(stream.txbuff.write_available),
                                           nullout or str(stream.rxbuff.read_available),
                                           stream.xruns, 100*stream.cpu_load)
                    _sys.stdout.write(line); _sys.stdout.flush()
        finally:
            print()
    except AudioBufferError as buffexc:
        print("AudioBufferError:", buffexc, file=_sys.stderr)
    except KeyboardInterrupt:
        pass

    print("Callback info:")
    print("\tFrames processed: %d ( %7.3fs )" % (stream.frame_count, stream.frame_count/float(stream.samplerate)))
    print("\txruns (under/over): input {0.inputUnderflows}/{0.inputOverflows}, output {0.outputUnderflows}/{0.outputOverflows}".format(stream._cstream))

    if _lib.PYPA_DEBUG:
        cbinfo = stream._callback_info
        print("\tCallback serviced %d times" % cbinfo.call_count)
        print("\tDelta range (ms): [ {:7.3f}, {:7.3f}]".format(1e3*(cbinfo.min_dt), 1e3*(cbinfo.max_dt)))
        print("\tNominal (ms): %s" %
              ('%7.3f' % (1e3*stream.blocksize/stream.samplerate) if stream.blocksize else 'N/A'))
        print("\tWrite/read misses: %d/%d" % (stream._wmisses, stream._rmisses))

        arr = filter(bool, _ffi.unpack(cbinfo.period, _lib.MEASURE_LEN))
        print(arr)
        print(min(arr), max(arr), sorted(arr)[len(arr)//2])

    return 0

if __name__ == '__main__':
    _sys.exit(_main())
