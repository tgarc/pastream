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
"""
pastream: Portaudio Streams for Python
"""
from __future__ import print_function as _print_function
try:                import Queue as _queue
except ImportError: import queue as _queue
try:                import numpy as _np
except ImportError: _np = None
import threading as _threading
import time as _time
import sys as _sys
import sounddevice as _sd
import soundfile as _sf
import weakref as _weakref
from _py_pastream import ffi as _ffi, lib as _lib


__version__ = '0.0.2'
__usage__ = "%(prog)s [options] [-d device] input output"

# Set a default size for the audio callback ring buffer
_PA_BUFFERSIZE = 1 << 20

# Private states that determine how a stream completed
_FINISHED = 1
_ABORTED = 2
_STOPPED = 4

# Include xrun flags in nampespace
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


# 'vendored' code from pa_ringbuffer:
#     https://github.com/mgeier/python-pa-ringbuffer
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


# TODO: add option to do asynchronous exception raising
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
    nframes : int, optional
        Number of frames to play/record. (0 means unlimited). This does *not*
        include the length of any additional padding.
    padding : int, optional
        Number of zero frames to pad the output with. This has no effect on the
        input.
    offset : int, optional
        Number of frames to discard from beginning of input. This has no effect
        on the output.
    buffersize : int, optional
        Transmit/receive buffer size in units of frames. Increase for smaller
        blocksizes.
    reader, writer : function or None, optional
        Buffer reader and writer functions to be run in a separate thread.
    raise_on_xruns : int or bool, optional
        Abort the stream on a particular xrun condition and raise an XRunError
        exception. Accepts a combination of
        pa{Input,Output}{Overflow,Underflow} or True (which indicates any xrun
        condition).
    blocksize : int, optional
        Portaudio buffer size. If None or 0, the Portaudio backend will
        automatically determine a size.

    Other Parameters
    ----------------
    kind, device, channels, dtype, **kwargs
        Additional parameters to pass to ``sounddevice._StreamBase``.

    Attributes
    ----------
    txbuff : Ringbuffer
        RingBuffer used for writing audio data to the output Portaudio stream.
    rxbuff : Ringbuffer
        RingBuffer used for reading audio data from the input Portaudio stream.
    """

    def __init__(self, kind, device=None, samplerate=None, channels=None,
                 dtype=None, nframes=0, padding=0, offset=0,
                 buffersize=_PA_BUFFERSIZE, reader=None, writer=None,
                 keep_alive=False, raise_on_xruns=False, blocksize=None,
                 **kwargs):
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
            self.txbuff = RingBuffer(isize * iparameters.channelCount, buffersize)
            self.rxbuff = RingBuffer(osize * oparameters.channelCount, buffersize)
        else:
            parameters, dtype, samplesize, _ = _sd._get_stream_parameters(
                kind, device, channels, dtype, None, None, None)
            framesize = samplesize * parameters.channelCount
            if kind == 'output':
                self.txbuff = RingBuffer(framesize, buffersize)
            if kind == 'input':
                self.rxbuff = RingBuffer(framesize, buffersize)

        # Set up the C portaudio callback
        self._cstream = _ffi.NULL
        self.__nframes = nframes
        self.__weakref = _weakref.WeakKeyDictionary()
        if kwargs.get('callback', None) is None:
            # Create the C BufferedStream object
            lastTime = _ffi.new('PaStreamCallbackTimeInfo*')
            self._cstream = _ffi.new("Py_PaBufferedStream*")
            self._cstream.lastTime = lastTime
            self.__weakref[self._cstream] = lastTime

            # Init the C BufferedStream object
            if raise_on_xruns is True: raise_on_xruns = 0xF
            _lib.init_stream(self._cstream, int(keep_alive),
                int(raise_on_xruns), nframes, padding, offset,
                _ffi.NULL, _ffi.NULL)

            # Cast our ring buffers for use in C
            if self.rxbuff is not None: self._cstream.rxbuff = \
               _ffi.cast('PaUtilRingBuffer*', self.rxbuff._ptr)
            if self.txbuff is not None: self._cstream.txbuff = \
               _ffi.cast('PaUtilRingBuffer*', self.txbuff._ptr)

            # Pass our data and callback to sounddevice
            kwargs['userdata'] = self._cstream
            kwargs['callback'] = _ffi.addressof(_lib, 'callback')
            kwargs['wrap_callback'] = None

        # These flags are used to tell when the callbacks have finished. We can
        # use them to abort writing of the ringbuffer.
        self.__statecond = _threading.Condition()
        self.__streamlock = _threading.RLock()
        self.__state = 0
        self.__aborting = False
        self.__exceptions = _queue.Queue()

        # Set up reader/writer threads
        self._owner_thread = _threading.current_thread()
        self._rxthread = self._txthread = None
        if (kind == 'duplex' or kind == 'output') and writer is not None:
            self._txthread_args = { 'target': self._readwritewrapper,
                                    'args': (self.txbuff, writer) }
        else:
            self._txthread_args = None
        if (kind == 'duplex' or kind == 'input') and reader is not None:
            self._rxthread_args = { 'target': self._readwritewrapper,
                                    'args': (self.rxbuff, reader) }
        else:
            self._rxthread_args = None

        # TODO?: add support for C finished_callback function pointer
        user_callback = kwargs.pop('finished_callback', lambda : None)

        def finished_callback():
            # Check for any errors that might've occurred in the callback
            msg = _ffi.string(self._cstream.errorMsg).decode('utf-8')
            if len(msg):
                exctype, excmsg = msg.split(':', 1) if ':' in msg else (msg, '')
                exctype = getattr(_sys.modules[__name__], exctype)
                if exctype is XRunError and not len(excmsg):
                    excmsg = str(_sd.CallbackFlags(self.status))
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
                self.__statecond.notify_all()

            user_callback()

        super(_BufferedStreamBase, self).__init__(kind, blocksize=blocksize,
            device=device, samplerate=samplerate, channels=channels,
            dtype=dtype, finished_callback=finished_callback, **kwargs)

        # DEBUG for measuring polling performance
        self._rmisses = self._wmisses = 0

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
        Block until stream state changes to finished/aborted/stopped or until
        the optional timeout occurs.

        Returns
        -------
        bool
        True unless the timeout occurs.

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
        return self._cstream.status

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
        Total number of frames to be processed by the stream. A value of zero
        means no limit.
        """
        # We fib a bit here: __nframes != _cstream.nframes. The former doesn't
        # include any padding.
        return self.__nframes

    @nframes.setter
    def nframes(self, value):
        # Set nframes in an atomic manner
        self._cstream.nframes = value + self._cstream.padding if value else 0
        self.__nframes = value

    def __enter__(self):
        return self

    def __exit__(self, exctype, excvalue, exctb):
        self.close()

    def _prepare(self):
        assert not self.active, "Stream has already been started!"

        # Apparently when using a PaStreamFinishedCallback the stream
        # *must* be stopped before starting the stream again or the
        # streamFinishedCallback will never be called
        super(_BufferedStreamBase, self).stop()

        # Reset stream state machine
        with self.__statecond:
            self.__state = 0
        with self.__streamlock:
            self.__aborting = False

        # Reset c stream info
        _lib.reset_stream(self._cstream)
        self.nframes = self.__nframes

        # Clear any rx buffer data still leftover
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
        if isinstance(self.device, int) or self.device[0] == self.device[1]:
            name = _sd.query_devices(self._device)['name']
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
        return ("{0.__name__}('{1}', samplerate={2._samplerate:.0f}, "
                "channels={3}, dtype='{4}', blocksize={2._blocksize})").format(
            self.__class__, name, self, channels, dtype)


# Mix-in purely for adding chunks method
class _InputStreamMixin(object):

    # TODO: add buffer 'type' as an argument
    # TODO: add fill_value option
    def chunks(self, chunksize=None, overlap=0, always_2d=False, out=None):
        """
        Similar in concept to the PySoundFile library's `blocks`
        method. Returns an iterator over buffered audio chunks read
        from a Portaudio stream.

        Parameters
        ----------
        chunksize : int, optional
            Size of iterator chunks. If not specified, the estimated
            stream input latency will be used to determine the chunk
            size.
        overlap : int, optional
            Number of frames to overlap across blocks.
        always_2d : bool, optional
            Always returns blocks 2 dimensional arrays. Only valid if you have
            numpy installed.
        out : buffer object, optional
            If you would like use your own buffer-implementing object
            you can pass it here. Note this expects a single-byte
            elements as would be provided by e.g., bytearray

        See Also
        --------
        :func:`chunks`

        Yields
        ------
        array
            ndarray or memoryview object with `chunksize` elements.
        """
        try:
            channels = self.channels[0]
            latency = self.latency[0]
            dtype = self.dtype[0]
        except TypeError:
            latency = self.latency
            dtype = self.dtype
            channels = self.channels

        varsize = False
        if not chunksize:
            if self.blocksize:
                chunksize = self.blocksize
            else:
                varsize = True
                chunksize = int(round(latency * self.samplerate))
            if overlap:
                raise ValueError(
                    "Using overlap requires a fixed chunksize or stream blocksize")
        if overlap >= chunksize:
            raise ValueError(
                "Overlap must be less than chunksize or stream blocksize")
        if always_2d and (_np is None or out is not None):
            raise ValueError("always_2d is only supported with numpy arrays")

        if out is not None:
            bytebuff = tempbuff = out
        elif _np is None:
            if varsize: nbytes = len(self.rxbuff) * self.rxbuff.elementsize
            else:       nbytes = chunksize * self.rxbuff.elementsize
            bytebuff = tempbuff = memoryview(bytearray(nbytes))
        else:
            if channels > 1: always_2d = True
            if varsize: nframes = len(self.rxbuff)
            else:       nframes = chunksize
            tempbuff = _np.zeros((nframes, channels) if always_2d
                                 else nframes * channels, dtype=dtype)
            try:                   bytebuff = tempbuff.data.cast('B')
            except AttributeError: bytebuff = tempbuff.data

        copy = bool(overlap) or out is not None
        incframes = chunksize - overlap
        rxbuff = self.rxbuff
        boverlap = overlap * rxbuff.elementsize
        minframes = 1 if varsize else incframes

        starttime = dt = rmisses = 0  # DEBUG
        wait_time = overtime = 0
        done = False
        self.start()
        try:
            sleeptime = (incframes - rxbuff.read_available) / self.samplerate
            if sleeptime > 0:
                _time.sleep(sleeptime)
            while not (self.aborted or done):
                # for thread safety, check the stream is active *before* reading
                active = self.active
                nframes = rxbuff.read_available
                lastTime = self._cstream.lastTime.currentTime
                if nframes < minframes:
                    if not wait_time:
                        wait_time = self.time
                    if not active:
                        done = True
                    else:
                        self._rmisses += 1
                        _time.sleep(0.0025)
                        continue
                elif wait_time:
                    overtime = wait_time - lastTime
                    wait_time = 0

                # Debugging only
                # print("{0:7.3f} {1:7.3f} {2:7.3f} {3:7.3f} {4} {5}".format(
                #     1e3*(_time.time() - starttime), 1e3*sleeptime, 1e3*overtime,
                #     1e3*dt, self._rmisses - rmisses, nframes - incframes))
                rmisses = self._rmisses
                starttime = _time.time()

                nframes, buffregn1, buffregn2 = rxbuff.get_read_buffers(
                    nframes if varsize else incframes)

                if copy or len(buffregn2):
                    n2offset = boverlap + len(buffregn1)
                    bytebuff[boverlap:n2offset] = buffregn1
                    if len(buffregn2):
                        bytebuff[n2offset:n2offset + len(buffregn2)] = buffregn2
                    rxbuff.advance_read_index(nframes)
                    if _np:
                        yield tempbuff[:overlap + nframes]
                    else:
                        yield bytebuff[:boverlap + nframes * rxbuff.elementsize]
                    if overlap:
                        bytebuff[:boverlap] = bytebuff[-boverlap:]
                else:
                    if always_2d:
                        yield _np.frombuffer(buffregn1, dtype=dtype)\
                                 .reshape(nframes, channels)
                    elif _np:
                        yield _np.frombuffer(buffregn1, dtype=dtype)
                    else:
                        yield buffregn1
                    rxbuff.advance_read_index(nframes)

                dt = _time.time() - starttime  # DEBUG

                sleeptime = (incframes - rxbuff.read_available) / self.samplerate \
                    + self._cstream.lastTime.currentTime - self.time \
                    - overtime
                if sleeptime > 0:
                    _time.sleep(sleeptime)
        except:
            try:                       self.abort()
            except _sd.PortAudioError: pass
            raise


class BufferedInputStream(_InputStreamMixin, _BufferedStreamBase):
    def __init__(self, *args, **kwargs):
        super(BufferedInputStream, self).__init__('input', *args, **kwargs)


class BufferedOutputStream(_BufferedStreamBase):
    def __init__(self, *args, **kwargs):
        super(BufferedOutputStream, self).__init__('output', *args, **kwargs)


class BufferedStream(BufferedInputStream, BufferedOutputStream):
    def __init__(self, *args, **kwargs):
        _BufferedStreamBase.__init__(self, 'duplex', *args, **kwargs)


class _SoundFileStreamBase(_BufferedStreamBase):
    """
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
    format, subtype, endian
        Parameters to pass to SoundFile constructor(s). Accepts pairs to allow
        different parameters for input and output.
    reader, writer, kind, blocksize, **kwargs
        Additional parameters to pass to _BufferedStreamBase.
    """

    def __init__(self, kind, inpf=None, outf=None, reader=None, writer=None,
                 format=None, subtype=None, endian=None, **kwargs):
        iformat, oformat = _sd._split(format)
        isubtype, osubtype = _sd._split(subtype)
        iendian, oendian = _sd._split(endian)

        raw_input = iformat and iformat.lower() == 'raw'

        # At this point we don't care what 'kind' the stream is, only whether
        # the input/output is None which determines whether reader/writer
        # functions should be registered
        self._inpf = inpf
        if self._inpf is not None:
            if writer is None:
                writer = self._soundfilereader
            if isinstance(self._inpf, _sf.SoundFile):
                self.inp_fh = self._inpf
            elif not raw_input:
                self.inp_fh = _sf.SoundFile(self._inpf)
                if kwargs.get('samplerate', None) is None:
                    kwargs['samplerate'] = self.inp_fh.samplerate
                if kwargs.get('channels', None) is None:
                    kwargs['channels'] = self.inp_fh.channels
        else:
            self.inp_fh = self._inpf

        # We need to set the reader here; output file parameters will be known
        # once we open the stream
        self._outf = outf
        if outf is not None and reader is None:
            reader = self._soundfilewriter
        self.out_fh = None

        super(_SoundFileStreamBase, self).__init__(kind,
            reader=reader, writer=writer, **kwargs)

        # For raw input file, assume the format corresponds to the device
        # parameters
        if raw_input:
            channels = kind == 'duplex' and self.channels[1] or self.channels
            self.inp_fh = _sf.SoundFile(self._inpf, 'r', int(self.samplerate),
                              channels, isubtype, iendian, 'raw')

        # Try and determine the file extension here; we need to know it if we
        # want to try and set a default subtype for the output
        if not oformat:
            try:
                oformat = getattr(outf, 'name', outf).rsplit('.', 1)[1].lower()
            except (AttributeError, IndexError):
                pass

        # If the output file hasn't already been opened, we open it here using
        # the input file and output stream settings, plus any user supplied
        # arguments
        if not (self._outf is None or isinstance(self._outf, _sf.SoundFile)):
            # For those file formats which support PCM or FLOAT, use the device
            # samplesize to make a guess at a default subtype
            dtype, ssize, channels = self.dtype, self.samplesize, self.channels
            if kind == 'duplex':
                channels = channels[0]; dtype = dtype[0]; ssize = ssize[0]
            if 'float' in dtype: subtype = 'float'
            else:                subtype = 'pcm_{0}'.format(8 * ssize)
            if osubtype is None and oformat:
                if _sf.check_format(oformat, subtype, oendian):
                    osubtype = subtype
                else:
                    raise ValueError('''\
Could not determine an appropriate default subtype for '{0}' output file
format: please specify subtype'''.format(oformat))
            self.out_fh = _sf.SoundFile(self._outf, 'w', int(self.samplerate),
                              channels, osubtype, oendian, oformat)
        else:
            self.out_fh = self._outf

    # Default handler for writing input from a BufferedStream to a SoundFile
    # object
    @staticmethod
    def _soundfilewriter(stream, rxbuff):
        if stream.txbuff is not None:
            dtype = stream.dtype[0]
        else:
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

    # Default handler for reading input from a SoundFile object and writing it
    # to a BufferedStream
    @staticmethod
    def _soundfilereader(stream, txbuff):
        if stream.rxbuff is not None:
            dtype = stream.dtype[1]
        else:
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

            # exit if we've reached end of file
            if readframes < nframes: break

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
        Output file to write captured audio data. If a SoundFile is not passed,
        the output file parameters will be determined from the output audio
        device stream.
    format, subtype, endian
        Arguments to pass to the SoundFile constructor when outf is not already
        a SoundFile object.

    Other Parameters
    -----------------
    **kwargs
        Additional parameters to pass to SoundFileStreamBase.
    """

    def __init__(self, outf, **kwargs):
        super(SoundFileInputStream, self).__init__('input', outf=outf,
                                                   **kwargs)


class SoundFileOutputStream(_SoundFileStreamBase):
    """
    Audio file player.

    Parameters
    -----------
    inpf : SoundFile compatible input
        Input file to stream to audio device. Except in the case of a RAW file,
        the audio file will determine the default ``samplerate`` and
        ``channels`` for the output audio stream.
    """

    def __init__(self, inpf, buffersize=_PA_BUFFERSIZE, **kwargs):
        super(SoundFileOutputStream, self).__init__('output', inpf=inpf,
            buffersize=buffersize, **kwargs)


class SoundFileStream(SoundFileInputStream, SoundFileOutputStream):
    """
    Full duplex audio file streamer. Note that only one of inpf and outf
    is required. This allows you to e.g. use a SoundFile as input but
    implement your own reader and/or read from the buffer in the
    stream's owner thread.
    """

    def __init__(self, inpf=None, outf=None, buffersize=_PA_BUFFERSIZE, **kwargs):
        # If you're not using soundfiles for the input or the output,
        # then you should probably be using the BufferedStream class
        if inpf is None and outf is None:
            raise ValueError("No input or output file given.")

        _SoundFileStreamBase.__init__(self, 'duplex', inpf=inpf, outf=outf,
            buffersize=buffersize, **kwargs)

# TODO: add support for generic 'input' which accepts file, buffer, or ndarray


def chunks(chunksize=None, overlap=0, always_2d=False, out=None, inpf=None,
           streamclass=None, **kwargs):
    """
    Read audio data in iterable chunks from a Portaudio stream.

    Parameters
    ------------
    chunksize, overlap, always_2d, out
        See :meth:`BufferedStream.chunks` for description.
    inpf : SoundFile compatible input, optional
        Optional input stimuli.

    Other Parameters
    -----------------
    streamclass : object
        Base class to use. By default the streamclass will be one of
        BufferedInputStream, BufferedStream or SoundFileStream depending
        on whether an input file and/or `writer` argument was supplied.
    **kwargs
        Additional arguments to pass to base stream class.

    See Also
    --------
    :meth:`BufferedInputStream.chunks`, :meth:`BufferedStream.chunks`

    Yields
    -------
    array
        ndarray or memoryview object with `chunksize` elements.
    """
    if streamclass is None:
        if inpf is not None:
            stream = SoundFileStream(inpf, **kwargs)
        elif kwargs.get('writer', None) is not None:
            stream = BufferedStream(**kwargs)
        else:
            stream = BufferedInputStream(**kwargs)
    elif inpf is None:
        stream = streamclass(**kwargs)
    else:
        stream = streamclass(inpf, **kwargs)

    try:
        for blk in stream.chunks(chunksize, overlap, always_2d, out):
            yield blk
    finally:
        try:                       stream.close()
        except _sd.PortAudioError: pass


# Used solely for the pastream app
def _SoundFileStreamFactory(inpf=None, outf=None, **kwargs):
    if inpf is not None and outf is not None:
        Streamer = SoundFileStream
        kind = 'duplex'; ioargs = (inpf, outf)
    elif inpf is not None:
        Streamer = SoundFileOutputStream
        kind = 'input'; ioargs = (inpf,)
    elif outf is not None:
        Streamer = SoundFileInputStream
        kind = 'output'; ioargs = (outf,)
    else:
        raise SystemExit("No input or output selected.")

    return Streamer(*ioargs, **kwargs), kind


def _get_parser(parser=None):
    from argparse import Action, ArgumentParser, RawDescriptionHelpFormatter

    if parser is None:
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                                fromfile_prefix_chars='@', usage=__usage__,
                                description='''\
Cross platform audio playback and capture.''')
        parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()

    class ListStreamsAction(Action):
        def __call__(*args, **kwargs):
            print(_sd.query_devices())
            _sys.exit(0)

    def csv(arg):
        subtype = arg if callable(arg) else (lambda x: x)
        def csv(value):
            values = [subtype(v) or None for v in value.split(',')]
            return values[0] if len(values) == 1 else values
        return csv if callable(arg) else csv(arg)

    def dvctype(dvc):
        dvclist = []
        for v in dvc.split(','):
            try:               dvclist.append(int(v))
            except ValueError: dvclist.append(v or None)
        return dvclist[0] if len(dvclist) == 1 else dvclist

    def sizetype(x):
        if x.endswith('k'):   x = int(float(x[:-1]) * 1e3)
        elif x.endswith('K'): x = int(float(x[:-1]) * 1024)
        elif x.endswith('m'): x = int(float(x[:-1]) * 1e6)
        elif x.endswith('M'): x = int(float(x[:-1]) * 1024 * 1024)
        else:                 x = int(x)
        assert x > 0, "Must be a positive value."
        return x

    def posint(intarg):
        intarg = int(intarg)
        assert intarg > 0, "Must be a positive value."
        return intarg

    parser.add_argument("input", type=lambda x: None if x == 'null' else x,
        help='''\
Input audio file. Use the special designator 'null' for recording only. A
single dash '-' may be used to read from STDIN.''')

    parser.add_argument("output", type=lambda x: None if x == 'null' else x,
        help='''\
Output audio file. Use the special designator 'null' for playback only. A
single dash '-' may be used to write to STDOUT.''')

#     parser.add_argument("--loop", default=False, nargs='?', metavar='n',
#                         const=True, type=int,
#                         help='''\
# Replay the playback file n times. If no argument is specified, playback will
# loop infinitely. Does nothing if there is no playback.''')

    parser.add_argument("-l", action=ListStreamsAction, nargs=0,
        help="List available audio device streams.")

    parser.add_argument("-q", "--buffersize", type=sizetype,
                        default=_PA_BUFFERSIZE, help='''\
File buffering size (in units of frames). Must be a power of
2. Determines the maximum amount of buffering for the input/output
file(s). Use higher values to increase robustness against irregular
file i/o behavior. Add a 'K' or 'M' suffix to specify size in kibi or
mebi respectively. (Default 0x%(default)x)''')

    parser.add_argument("-p", "--pad", type=sizetype, default=0, help='''\
Pad the input with frames of zeros. Useful to avoid truncating full duplex
recording.''')

    parser.add_argument("-o", "--offset", type=sizetype, default=0, help='''\
Drop a number of frames from the start of a recording.''')

    parser.add_argument("-n", "--nframes", type=sizetype, default=0, help='''\
Limit playback/capture to this many frames.''')

    devopts = parser.add_argument_group("I/O device stream options")

    devopts.add_argument("-d", "--device", type=dvctype,
        help='''\
Audio device name expression or index number. Defaults to the
PortAudio default device.''')

    devopts.add_argument("-b", "--blocksize", type=posint, help='''\
PortAudio buffer size in units of frames. If zero or not specified, backend
will decide an optimal size.''')

    devopts.add_argument("-f", "--format", dest='dtype',
        type=csv, choices=_sd._sampleformats.keys(), help='''\
Sample format of device I/O stream.''')

    devopts.add_argument("-c", "--channels", type=csv(int),
        help="Number of channels.")

    devopts.add_argument("-r", "--rate", dest='samplerate',
        type=lambda x: int(float(x[:-1]) * 1000) if x.endswith('k') else int(x),
        help='''\
Sample rate in Hz. Add a 'k' suffix to specify kHz.''')

    fileopts = parser.add_argument_group('''\
Audio file formatting options. Options accept single values or pairs''')

    fileopts.add_argument("-t", dest="file_type", type=csv(str.upper),
        choices=_sf.available_formats().keys(), help='''\
Audio file type. (Required for RAW files). Typically this is determined from
the file header or extension, but it can be manually specified here.''')

    fileopts.add_argument("-e", dest="encoding", type=csv(str.upper),
        choices=_sf.available_subtypes(), help='''\
Sample format encoding. Note for output file encodings: for file types that
support PCM or FLOAT format, pastream will automatically choose the sample
format that best matches the output device stream; otherwise, the subtype is
required.''')

    fileopts.add_argument("--endian", type=csv(str.lower),
        choices=['file', 'big', 'little'], help="Sample endianness.")

    return parser


def _main(argv=None):
    from itertools import chain
    if argv is None: argv = _sys.argv[1:]
    parser = _get_parser()
    args = parser.parse_args(argv)

    stream, kind = _SoundFileStreamFactory(args.input, args.output,
                        samplerate=args.samplerate,
                        blocksize=args.blocksize,
                        buffersize=args.buffersize,
                        nframes=args.nframes, padding=args.pad,
                        offset=args.offset, endian=args.endian,
                        subtype=args.encoding, format=args.file_type,
                        device=args.device, channels=args.channels,
                        dtype=args.dtype)

    statline = "\r{:8.3f}s {:10d} frames processed, {:>8s} frames free, " \
               "{:>8s} frames queued ({:d} xruns, {:f}% load)\r"
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
                    _time.sleep(0.15)
                    line = statline.format(_time.time() - t1, stream.frame_count,
                                           nullinp
                                           or str(stream.txbuff.write_available),
                                           nullout
                                           or str(stream.rxbuff.read_available),
                                           stream.xruns, 100 * stream.cpu_load)
                    _sys.stdout.write(line); _sys.stdout.flush()
        finally:
            print()
    except KeyboardInterrupt:
        pass

    print("Callback info:")
    print("\tFrames processed: %d ( %7.3fs )"
          % (stream.frame_count, stream.frame_count / float(stream.samplerate)))
    print('''\
\txruns (under/over): input {0.inputUnderflows}/{0.inputOverflows}
\txruns (under/over): output {0.outputUnderflows}/{0.outputOverflows}'''
          .format(stream._cstream))
    print("\tCallback serviced %d times" % stream._cstream.call_count)
    print("\tWrite/read misses: %d/%d" % (stream._wmisses, stream._rmisses))

    return 0


if __name__ == '__main__':
    _sys.exit(_main())
