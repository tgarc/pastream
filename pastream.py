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
try:                import numpy as _np
except ImportError: _np = None
import threading as _threading
import time as _time
import sys as _sys
import sounddevice as _sd
import soundfile as _sf
from _py_pastream import ffi as _ffi, lib as _lib
from pa_ringbuffer import init as _ringbuffer_init


__version__ = '0.0.8'
__usage__ = "%(prog)s [options] input output"


LOOP = False

# Set a default size for the audio callback ring buffer
_PA_BUFFERSIZE = 1 << 16

# Private states that determine how a stream completed
_FINISHED = 1
_ABORTED = 2
_STOPPED = 4

# Include xrun flags in nampespace
paInputOverflow = _lib.paInputOverflow
paInputUnderflow = _lib.paInputUnderflow
paOutputOverflow = _lib.paOutputOverflow
paOutputUnderflow = _lib.paOutputUnderflow


class BufferFull(Exception):
    pass


class BufferEmpty(Exception):
    pass


RingBuffer = _ringbuffer_init(_ffi, _lib)


# TODO?: add option to do asynchronous exception raising
class _StreamBase(_sd._StreamBase):
    """Abstract base stream class from which all other stream classes derive.

    Note that this class inherits from :mod:`sounddevice`'s ``_StreamBase``
    class.
    """
    def __init__(self, kind, device=None, samplerate=None, channels=None,
                 dtype=None, frames=-1, pad=0, offset=0,
                 buffersize=_PA_BUFFERSIZE, reader=None, writer=None,
                 blocksize=None, **kwargs):
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
        self.__frames = frames
        if kwargs.get('callback', None) is None:
            # Init the C PyPaStream object
            self._cstream = _ffi.new("Py_PaStream*")
            _lib.init_stream(self._cstream)

            self.frames = frames
            self.pad = pad
            self.offset = offset

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

        super(_StreamBase, self).__init__(kind, blocksize=blocksize,
            device=device, samplerate=samplerate, channels=channels,
            dtype=dtype, finished_callback=finished_callback, **kwargs)

        if kind == 'output' or kind == 'duplex':
            assert len(self.txbuff) >= self.blocksize, \
                "buffersize must be >= the audio device blocksize"
        if kind == 'input' or kind == 'duplex':
            assert len(self.rxbuff) >= self.blocksize, \
                "buffersize must be >= the audio device blocksize"

        # DEBUG for measuring polling performance
        self._rmisses = self._wmisses = 0

        self._autoclose = False

    def __stopiothreads(self):
        # !This function is *not* thread safe!
        currthread = _threading.current_thread()
        if self._rxthread is not None and self._rxthread.is_alive() \
           and self._rxthread != currthread:
            self._rxthread.join()
        if self._txthread is not None and self._txthread.is_alive() \
           and self._txthread != currthread:
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
    def offset(self):
        return self._cstream.offset

    @offset.setter
    def offset(self, value):
        self._cstream.offset = value

    @property
    def pad(self):
        return self._cstream.pad

    @pad.setter
    def pad(self, value):
        # Note that the py_pastream callback doesn't act on 'pad' unless
        # frames < 0; thus, set 'frames' first to get deterministic behavior.
        frames = self.__frames
        if frames >= 0 and value >= 0:
            self._cstream.frames = value + frames
        self._cstream.pad = value

    @property
    def frames(self):
        # We fib a bit here: __frames is _cstream.frames minus any padding
        return self.__frames

    @frames.setter
    def frames(self, value):
        pad = self._cstream.pad
        if value > 0 and pad > 0:
            self._cstream.frames = value + pad
        else:
            self._cstream.frames = value
        self.__frames = value

    def __enter__(self):
        return self

    def __exit__(self, exctype, excvalue, exctb):
        self.close()

    def _prepare(self):
        assert not self.active, "Stream has already been started!"

        # Apparently when using a PaStreamFinishedCallback the stream
        # *must* be stopped before starting the stream again or the
        # streamFinishedCallback will never be called
        if self.__state != 0 and not self.stopped:
            super(_StreamBase, self).stop()

        # Reset stream state machine
        with self.__statecond:
            self.__state = 0
        with self.__streamlock:
            self.__aborting = False

        # Reset cstream info
        _lib.reset_stream(self._cstream)

        # Recreate the necessary threads
        if self._rxthread_args is not None:
            self._rxthread = _threading.Thread(**self._rxthread_args)
            self._rxthread.daemon = True
        if self._txthread_args is not None:
            self._txthread = _threading.Thread(**self._txthread_args)
            self._txthread.daemon = True

    def start(self, prebuffer=True):
        '''Start the audio stream

        Parameters
        ----------
        prebuffer : bool, optional
            For threading only: wait for the first output write before starting
            the audio stream. If not using threads this has no effect.
        '''
        self._prepare()
        if self._txthread is not None:
            self._txthread.start()
            if prebuffer:
                while not self.txbuff.read_available and self._txthread.is_alive():
                    _time.sleep(0.005)
            self._reraise_exceptions()
        super(_StreamBase, self).start()
        if self._rxthread is not None:
            self._rxthread.start()
            self._reraise_exceptions()

    def stop(self):
        with self.__streamlock:
            super(_StreamBase, self).stop()
        self.__stopiothreads()
        self._reraise_exceptions()

    def abort(self):
        with self.__streamlock:
            self.__aborting = True
            super(_StreamBase, self).abort()
        self.__stopiothreads()
        self._reraise_exceptions()

    def close(self):
        # we take special care here to abort the stream first so that the
        # pastream pointer is still valid for the lifetime of the read/write
        # threads
        if not self.finished:
            with self.__streamlock:
                self.__aborting = True
                super(_StreamBase, self).abort()
        self.__stopiothreads()
        with self.__streamlock:
            super(_StreamBase, self).close()
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


# Mix-in purely for adding chunks method
class _InputStreamMixin(object):

    def chunks(self, chunksize=None, overlap=0, frames=None, always_2d=False, out=None):
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
        always_2d : bool, optional
            Always return chunks as 2 dimensional arrays. Only valid if you
            have numpy installed.
        out : :class:`~numpy.ndarray` or buffer object, optional
            Alternative output buffer in which to store the result. Note that
            any buffer object - with the exception of :class:`~numpy.ndarray` - is
            expected to have single-byte elements as would be provided by e.g.,
            ``bytearray``. ``bytes`` objects are not recommended as they will
            incur extra copies (use ``bytearray`` instead).

        Yields
        ------
        :class:`~numpy.ndarray`, buffer
            Buffer object with `chunksize` frames. If numpy is available
            defaults to :class:`~numpy.ndarray` otherwise a buffer of bytes is
            yielded (which is either a :class:`cffi.buffer` object or a
            ``memoryview``).

        """
        try:
            channels = self.channels[0]
            latency = self.latency[0]
            dtype = self.dtype[0]
        except TypeError:
            latency = self.latency
            dtype = self.dtype
            channels = self.channels

        rxbuff = self.rxbuff
        varsize = False
        if not chunksize:
            if self.blocksize:
                chunksize = self.blocksize + overlap
            elif overlap:
                raise ValueError(
                    "Using overlap requires a non-zero chunksize or stream blocksize")
            else:
                varsize = True
                chunksize = int(round(latency * self.samplerate))

        if overlap > chunksize:
            raise ValueError(
                "Overlap must be less than chunksize or stream blocksize")

        if always_2d and (_np is None or not isinstance(out, _np.ndarray)):
            raise ValueError("always_2d is only supported with numpy arrays")

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
            if varsize: nbytes = len(rxbuff) * rxbuff.elementsize
            else:       nbytes = chunksize * rxbuff.elementsize
            bytebuff = tempbuff = memoryview(bytearray(nbytes))
        else:
            numpy = True
            if channels > 1: always_2d = True
            if varsize: nframes = len(rxbuff)
            else:       nframes = chunksize
            tempbuff = _np.zeros((nframes, channels) if always_2d
                                 else nframes * channels, dtype=dtype)
            try:                   bytebuff = tempbuff.data.cast('B')
            except AttributeError: bytebuff = tempbuff.data

        boverlap = overlap * rxbuff.elementsize
        minframes = 1 if varsize else chunksize

        # fill the first overlap block with zeros
        if overlap:
            rxbuff.write(bytearray(boverlap))

        # DEBUG
        # logf = open('chunks2.log', 'wt')
        # print("delta sleep lag yield misses frames", file=logf)
        # starttime = dt = rmisses = 0

        wait_time = leadtime = 0
        done = False
        starttime = _time.time()
        if frames is not None:
            self.frames = frames

        self.start()
        try:
            sleeptime = latency - _time.time() + starttime - rxbuff.read_available / self.samplerate
            if sleeptime > 0:
                _time.sleep(max(self.offset / self.samplerate, sleeptime))
            while not (self.aborted or done):
                # for thread safety, check the stream is active *before* reading
                active = self.active
                frames = rxbuff.read_available
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

                frames, buffregn1, buffregn2 = rxbuff.get_read_buffers(
                    frames if varsize else chunksize)

                if out is not None or len(buffregn2):
                    buffsz1 = len(buffregn1)
                    bytebuff[:buffsz1] = buffregn1
                    bytebuff[buffsz1:buffsz1 + len(buffregn2)] = buffregn2
                    rxbuff.advance_read_index(frames - overlap)
                    if numpy:
                        yield tempbuff[:frames]
                    else:
                        yield bytebuff[:frames * rxbuff.elementsize]
                else:
                    if always_2d:
                        yield _np.frombuffer(buffregn1, dtype=dtype)\
                                 .reshape(frames, channels)
                    elif numpy:
                        yield _np.frombuffer(buffregn1, dtype=dtype)
                    else:
                        yield buffregn1
                    rxbuff.advance_read_index(frames - overlap)

                # # DEBUG
                # dt = _time.time() - starttime

                sleeptime = (chunksize - rxbuff.read_available) / self.samplerate \
                    + self._cstream.lastTime.currentTime - self.time              \
                    + leadtime
                if sleeptime > 0:
                    _time.sleep(sleeptime)
        except Exception:
            if _sd._initialized:
                self.stop()
                if self._autoclose: self.close()
            raise


class InputStream(_InputStreamMixin, _StreamBase):
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

    Attributes
    ----------
    rxbuff : :class:`Ringbuffer`
        RingBuffer used for storing data read from the audio device.

    """

    def __init__(self, *args, **kwargs):
        super(InputStream, self).__init__('input', *args, **kwargs)


class OutputStream(_StreamBase):
    """Playback only stream.

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
    writer : function, optional
        Dedicated function for feeding the output ring buffer.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to :class:`_StreamBase`.

    Attributes
    ----------
    txbuff : :class:`Ringbuffer`
        RingBuffer used for storing data to output to the audio device.

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
    buffersize : int, optional
        Transmit/receive buffer size in units of frames.
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
        Additional parameters to pass to :class:`_StreamBase`.

    Attributes
    ----------
    txbuff, rxbuff : :class:`RingBuffer`
        See :class:`OutputStream`, :class:`InputStream` for explanation.

    See Also
    --------
    :class:`OutputStream`, :class:`InputStream`
    """

    def __init__(self, *args, **kwargs):
        _StreamBase.__init__(self, 'duplex', *args, **kwargs)


# Helper function for SoundFileStream classes
def _soundfile_from_stream(stream, kind, file, mode, **kwargs):
    # Try and determine the file extension here; we need to know if we
    # want to try and set a default subtype for the file
    fformat = kwargs.pop('format', None)
    if not fformat:
        try:
            fformat = getattr(file, 'name', file).rsplit('.', 1)[1].lower()
        except (AttributeError, IndexError):
            fformat = None

    subtype = kwargs.pop('subtype', None)
    endian = kwargs.get('endian', None)
    if not subtype:
        # For those file formats which support PCM or FLOAT, use the device
        # samplesize to make a guess at a default subtype
        dtype = stream.dtype; ssize = stream.samplesize; channels = stream.channels
        if kind == 'duplex':
            channels = channels[0]; dtype = dtype[0]; ssize = ssize[0]
        if dtype == 'float32' and _sf.check_format(fformat, 'float', endian):
            subtype = 'float'
        else:
            subtype = 'pcm_{0}'.format(8 * ssize)
        if fformat and not _sf.check_format(fformat, subtype, endian):
            raise ValueError("Could not map stream datatype '{0}' to "
                "an appropriate subtype for '{1}' format; please specify"
                .format(dtype, fformat))

    if 'channels' not in kwargs:
        kwargs['channels'] = kind == 'duplex' and stream.channels['w' in mode] \
            or stream.channels

    return _sf.SoundFile(file, mode, int(stream.samplerate), subtype=subtype,
        format=fformat, **kwargs)


class _SoundFileStreamBase(_StreamBase):
    # See SoundFileStream for docstring
    def __init__(self, kind, inpf=None, outf=None, reader=None, writer=None,
                 format=None, subtype=None, endian=None, **kwargs):
        channels = kwargs.pop('channels', None)
        samplerate = kwargs.pop('samplerate', None)
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

        self._outf = outf
        if outf is not None:
            if writer is None:
                writer = self._soundfileplayer
            if not oformat and not raw_output:
                try:
                    raw_output = getattr(outf, 'name', outf)\
                        .rsplit('.', 1)[1].lower() == 'raw'
                except (AttributeError, IndexError):
                    pass
            if isinstance(outf, _sf.SoundFile):
                self.out_fh = outf
            elif not raw_output:
                self.out_fh = _sf.SoundFile(outf)
            elif not (samplerate and ochannels and osubtype):
                raise ValueError(
                    "samplerate, channels, and subtype must be specified for "
                    "RAW playback files")
            if samplerate is None:
                samplerate = self.out_fh.samplerate
            if ochannels is None:
                ochannels = self.out_fh.channels
                if kind == 'duplex':
                    channels = (ichannels or self.out_fh.channels, ochannels)
                else:
                    channels = self.out_fh.channels
        else:
            self.out_fh = outf

        if isinstance(inpf, _sf.SoundFile):
            if samplerate is None:
                samplerate = inpf.samplerate
            if ichannels is None:
                if kind == 'duplex':
                    channels = (inpf.channels, ochannels)
                else:
                    channels = inpf.channels

        # We need to set the reader here; recording file parameters will be known
        # once we open the stream
        self._inpf = inpf
        self.inp_fh = None
        if inpf is not None and reader is None:
            reader = self._soundfilerecorder

        super(_SoundFileStreamBase, self).__init__(kind, reader=reader,
            writer=writer, samplerate=samplerate, channels=channels,
            **kwargs)

        if raw_output:
            self.out_fh = _soundfile_from_stream(self, kind, outf, mode='r',
                subtype=osubtype, format='raw', endian=oendian)

        # If the recording file hasn't already been opened, we open it here using
        # the input file and stream settings, plus any user supplied arguments
        if inpf is not None and not isinstance(inpf, _sf.SoundFile):
            self.inp_fh = _soundfile_from_stream(self, kind, inpf, mode='w',
                subtype=isubtype, format=iformat, endian=iendian)
        else:
            self.inp_fh = self._inpf

    # Default handler for writing input from a Stream to a SoundFile
    # object
    @staticmethod
    def _soundfilerecorder(stream, rxbuff):
        inp_fh = stream.inp_fh
        if stream.txbuff is not None:
            dtype = stream.dtype[0]
        else:
            dtype = stream.dtype

        chunksize = min(8192, len(rxbuff))
        sleeptime = (chunksize - rxbuff.read_available) / stream.samplerate
        if sleeptime > 0:
            _time.sleep(max(sleeptime, stream.offset / stream.samplerate))
        while not stream.aborted:
            # for thread safety, check the stream is active *before* reading
            active = stream.active
            frames = rxbuff.read_available
            if frames == 0:
                # we've read everything and the stream is done; seeya!
                if not active: break
                # we're reading too fast, wait for a buffer write
                stream._rmisses += 1
                _time.sleep(0.0025)
                continue

            frames, buffregn1, buffregn2 = rxbuff.get_read_buffers(frames)
            inp_fh.buffer_write(buffregn1, dtype=dtype)
            if len(buffregn2):
                inp_fh.buffer_write(buffregn2, dtype=dtype)
            rxbuff.advance_read_index(frames)

            sleeptime = (chunksize - rxbuff.read_available) / stream.samplerate
            if sleeptime > 0:
                _time.sleep(sleeptime)

    # Default handler for reading input from a SoundFile object and writing it
    # to a Stream
    @staticmethod
    def _soundfileplayer(stream, txbuff):
        out_fh = stream.out_fh
        readinto = out_fh.buffer_read_into
        if stream.rxbuff is not None:
            dtype = stream.dtype[1]
        else:
            dtype = stream.dtype

        ptr1 = _ffi.new('void**')
        ptr2 = _ffi.new('void**')
        size1 = _ffi.new('ring_buffer_size_t*')
        size2 = _ffi.new('ring_buffer_size_t*')

        chunksize = min(8192, len(txbuff))
        sleeptime = (chunksize - txbuff.write_available) / stream.samplerate
        if sleeptime > 0:
            _time.sleep(sleeptime)
        while not stream.finished:
            frames = txbuff.write_available
            if not frames:
                stream._wmisses += 1
                stream.wait(0.0025)
                continue

            frames = _lib.PaUtil_GetRingBufferWriteRegions(
                txbuff._ptr, frames, ptr1, size1, ptr2, size2)
            buffregn1 = _ffi.buffer(ptr1[0], size1[0] * txbuff.elementsize)
            buffregn2 = _ffi.buffer(ptr2[0], size2[0] * txbuff.elementsize)

            readframes = readinto(buffregn1, dtype=dtype)
            if len(buffregn2):
                readframes += readinto(buffregn2, dtype=dtype)

            if LOOP:
                bytesz1 = size1[0] * txbuff.elementsize
                bytesz2 = size2[0] * txbuff.elementsize
                while readframes < frames:
                    out_fh.seek(0)
                    readbytes = readframes * txbuff.elementsize
                    if readbytes < bytesz1:
                        buffregn1 = _ffi.buffer(ptr1[0] + readbytes,
                                        bytesz1 - readbytes)
                        readframes += readinto(buffregn1, dtype=dtype)
                    else:
                        buffregn2 = _ffi.buffer(ptr2[0] + readbytes - bytesz1,
                                        bytesz2 + bytesz1 - readbytes)
                        readframes += readinto(buffregn2, dtype=dtype)
            txbuff.advance_write_index(readframes)
            if readframes < frames:
                break

            sleeptime = (chunksize - txbuff.write_available) / stream.samplerate
            if sleeptime > 0:
                _time.sleep(sleeptime)

    def close(self):
        try:
            super(_SoundFileStreamBase, self).close()
        finally:
            if not (self._outf is None or isinstance(self._outf, _sf.SoundFile)):
                self.out_fh.close()
            if not (self._inpf is None or isinstance(self._inpf, _sf.SoundFile)):
                self.inp_fh.close()


class SoundFileInputStream(_InputStreamMixin, _SoundFileStreamBase):
    """Audio file recorder.

    See :class:`SoundFileStream` for explanation of parameters.

    Parameters
    ----------
    inpf : :class:`~soundfile.SoundFile`, str, or file-like
        Input file to capture data from audio device. If a SoundFile is not
        passed, the input file parameters will be determined from the input
        audio stream.

    Attributes
    ------------
    inp_fh : :class:`~soundfile.SoundFile`
        The file object to capture data from the input ring buffer.

    """

    def __init__(self, inpf, **kwargs):
        super(SoundFileInputStream, self).__init__('input', inpf=inpf, **kwargs)


class SoundFileOutputStream(_SoundFileStreamBase):
    """Audio file player.

    See :class:`SoundFileStream` for explanation of parameters.

    Parameters
    ----------
    outf : :class:`~soundfile.SoundFile`, str, or file-like
        Output file to stream to audio device. The output file will determine
        the samplerate and number of channels for the audio stream.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to :class:`_StreamBase`.

    Attributes
    ------------
    out_fh : :class:`~soundfile.SoundFile`
        The file object to write to the output ring buffer.

    """

    def __init__(self, outf, buffersize=_PA_BUFFERSIZE, **kwargs):
        super(SoundFileOutputStream, self).__init__('output', outf=outf,
            buffersize=buffersize, **kwargs)


class SoundFileDuplexStream(SoundFileInputStream, SoundFileOutputStream):
    """Full duplex audio file streamer.
    Note that only one of inpf and outf is required. This allows you to
    e.g. use a SoundFile as input but implement your own reader and/or read
    from the buffer in the stream's owner thread.

    Other Parameters
    ----------------------
    inpf, outf
        See :class:`SoundFileInputStream`, :class:`SoundFileOutputStream`
    format, subtype, endian
        Parameters to pass to :class:`~soundfile.SoundFile`
        constructor(s). Accepts single values or pairs to allow different
        parameters for input and output files.
    **kwargs
        Additional parameters to pass to _StreamBase.
    """

    def __init__(self, inpf=None, outf=None, buffersize=_PA_BUFFERSIZE, **kwargs):
        # If you're not using soundfiles for the input or the output,
        # then you should probably be using the StreamBase class
        if inpf is None and outf is None:
            raise ValueError("No input or output file given.")

        _SoundFileStreamBase.__init__(self, 'duplex', inpf=inpf, outf=outf,
            buffersize=buffersize, **kwargs)


# TODO: add support for generic 'playback' which accepts soundfile, buffer, or
# ndarray
def chunks(chunksize=None, overlap=0, frames=-1, always_2d=False, out=None,
           playback=None, **kwargs):
    """Read audio data in iterable chunks from a Portaudio stream.

    Parameters
    ------------
    chunksize, overlap, frames, always_2d, out
        See :meth:`InputStream.chunks` for description.
    playback : :class:`~soundfile.SoundFile` compatible object, optional
        Optional playback file.

    Other Parameters
    -----------------
    **kwargs
        Additional arguments to pass to :class:`_StreamBase`.

    Yields
    -------
    buffer
        :class:`~numpy.ndarray` or memoryview object with `chunksize` elements.

    See Also
    --------
    :meth:`InputStream.chunks`

    """
    if streamclass is None:
        if kwargs.get('writer', None) is not None:
            stream = DuplexStream(**kwargs)
        elif playback is not None:
            stream = SoundFileDuplexStream(outf=playback, **kwargs)
        else:
            stream = InputStream(**kwargs)
    elif playback is None:
        stream = streamclass(**kwargs)
    else:
        stream = streamclass(playback, **kwargs)
    stream._autoclose = True
    return stream.chunks(chunksize, overlap, frames, always_2d, out)


# Used solely for the pastream app
def _SoundFileStreamFactory(inpf=None, outf=None, **kwargs):
    if inpf is not None and outf is not None:
        Streamer = SoundFileDuplexStream
        kind = 'duplex'; ioargs = (inpf, outf)
    elif outf is not None:
        Streamer = SoundFileOutputStream
        kind = 'output'; ioargs = (outf,)
    elif inpf is not None:
        Streamer = SoundFileInputStream
        kind = 'input'; ioargs = (inpf,)
    else:
        raise ValueError("At least one of {inpf, outf} must be specified.")

    return Streamer(*ioargs, **kwargs), kind


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
        if not x or x == 'null':
            return None
        return type(x) if type is not None else x

    def csvtype(arg, type=None):
        csvsplit = shlex.shlex(arg, posix=True)
        csvsplit.whitespace = ','; csvsplit.whitespace_split = True
        return list(map(type or str, csvsplit))

    parser.add_argument("input", type=nullortype,
        help='''\
Playback audio file. Use dash (-) to read from STDIN. Use 'null' or an empty
string ("") for record only.''')

    parser.add_argument("output", type=nullortype,
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

    propts.add_argument("-n", "--frames", type=sizetype, default=-1, help='''\
Limit playback/capture to this many frames. If FRAMES is negative (the
default), then streaming will continue until there is no playback data
remaining or, if no playback was given, recording will continue
indefinitely.''')

    propts.add_argument("--offset", type=possizetype, default=0, help='''\
Drop a number of frames from the start of a recording.''')

    propts.add_argument("-p", "--pad", type=sizetype, nargs='?', default=0,
        const=-1, help='''\
Pad the input with frames of zeros. (Useful to avoid truncating full duplex
recording). If PAD is negative (the default if no argument is given) then
padding is chosen so that the total playback length matches --frames. If frames
is also negative, zero padding will be added indefinitely.''')

    devopts = parser.add_argument_group("audio device options")

    devopts.add_argument("-b", "--blocksize", type=possizetype, help='''\
PortAudio buffer size in units of frames. If zero or not specified, backend
will decide an optimal size (recommended). ''')

    devopts.add_argument("-c", "--channels", type=lambda x: nullortype(x, int),
        nargs='+', help="Number of input/output channels.")

    devopts.add_argument("-d", "--device", type=lambda x: nullortype(x, dvctype),
        nargs='+', help='''\
Audio device name expression(s) or index number(s). Defaults to the
PortAudio default device(s).''')

    choices = list(_sd._sampleformats.keys())
    devopts.add_argument("-f", "--format", metavar="format", dest='dtype',
        type=nullortype, nargs='+', choices=choices + [None], help='''\
Sample format(s) of audio device stream. Must be one of {%s}.'''
% ', '.join(['null'] + choices))

    devopts.add_argument("-r", "--rate", dest='samplerate', type=possizetype,
        help='''\
Sample rate in Hz. Add a 'k' suffix to specify kHz.''')

    fileopts = parser.add_argument_group('''\
audio file formatting options. (options accept single values or pairs)''')

    choices = list(_sf.available_formats().keys())
    fileopts.add_argument("-t", "--file_type", metavar="file_type", nargs='+',
        type=lambda x: nullortype(x, str.upper), choices=choices + [None],
        help='''\
Audio file type(s). (Required for RAW files). Typically this is determined
from the file header or extension, but it can be manually specified here. Must
be one of {%s}.''' % ', '.join(['null'] + choices))

    choices = list(_sf.available_subtypes().keys())
    fileopts.add_argument("-e", "--encoding", metavar="encoding", nargs='+',
        type=lambda x: nullortype(x, str.upper), choices=choices + [None],
        help='''\
Sample format encoding(s). Note for output file encodings: for file types that
support PCM or FLOAT format, pastream will automatically choose the sample
format that most closely matches the output device stream; for other file
types, the subtype is required. Must be one of {%s}.'''
% ', '.join(['null'] + choices))

    choices = ['file', 'big', 'little']
    fileopts.add_argument("--endian", metavar="endian", nargs='+',
        type=lambda x: nullortype(x, str.lower), choices=choices + [None],
        help='''\
Sample endianness. Must be one of {%s}.''' % ', '.join(['null'] + choices))

    return parser


def _main(argv=None):
    import os, traceback

    if argv is None:
        argv = _sys.argv[1:]
    parser = _get_parser()
    args = parser.parse_args(argv)

    global LOOP
    LOOP = args.loop

    # Note that input/output from the cli perspective is reversed wrt the
    # pastream/portaudio library so we swap arguments here
    def unpack(x):
        return x[0] if x and len(x) == 1 else x and x[::-1]

    try:
        stream, kind = _SoundFileStreamFactory(args.output, args.input,
                           samplerate=args.samplerate,
                           blocksize=args.blocksize,
                           buffersize=args.buffersize, frames=args.frames,
                           pad=args.pad, offset=args.offset,
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

    nullinp = nullout = None
    if stream.out_fh is None:
        nullinp = 'n/a'
    elif args.loop and not stream.out_fh.seekable():
        raise ValueError("Can't loop playback; input is not seekable")
    if stream.inp_fh is None:
        nullout = 'n/a'
    elif stream.inp_fh.name == '-' or args.quiet:
        _sys.stdout = open(os.devnull, 'w')

    statline = "\r{:8.3f}s {:>8s} frames free, " \
               "{:>8s} frames queued ({:d} xruns, {:6.2f}% load)\r"
    print("-->", stream.out_fh if stream.out_fh is not None else 'null')
    print("<--", stream.inp_fh if stream.inp_fh is not None else 'null')
    print(["<->", "<--", "-->"][['duplex', 'output', 'input'].index(kind)], stream)

    with stream:
        try:
            stream.start()
            t1 = _time.time()
            while stream.active:
                line = statline.format(_time.time() - t1,
                                       nullinp
                                       or str(stream.txbuff.write_available),
                                       nullout
                                       or str(stream.rxbuff.read_available),
                                       stream.xruns, 100 * stream.cpu_load)
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
