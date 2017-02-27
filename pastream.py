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
try:
    _basestring = basestring
except NameError:
    _basestring = (str, bytes)
import threading as _threading
import time as _time
import sys as _sys
import sounddevice as _sd
import soundfile as _sf
import pa_ringbuffer as _pa_ringbuffer
from _pastream import ffi as _ffi, lib as _libps
try:
    import numpy as _np
except:
    _np = None
_libpa = _sd._lib
_ffirb = _pa_ringbuffer._ffi

__version__ = '0.0.0'
__usage__ = "%(prog)s [options] [-d device] input output"


PA_QSIZE = 1<<16 # Default number of frames to buffer i/o to portaudio callback


class AudioBufferError(Exception):
    pass

class _BufferedStreamBase(_sd._StreamBase):
    """
    This class adds a python Queue for reading and writing audio data. This
    double buffers the audio data so that any processing is kept out of the time
    sensitive audio callback function. For maximum flexibility, receive queue
    data is a bytearray object; transmit queue data should be of a buffer type
    where each element is a single byte.

    Notes:

    If the receive buffer fills or the transmit buffer is found empty during a
    callback the audio stream is aborted and an exception is raised.

    During playback, the end of the stream is signaled by an item on the queue
    that is smaller than blocksize*channels*samplesize bytes.

    Parameters
    -----------
    qsize : int
        Transmit/receive queue size in units of frames. Increase for smaller
        blocksizes.
    blocksize : int
        Portaudio buffer size. If None, the Portaudio backend will automatically
        determine a size.

    Other Parameters
    ----------------
    kind, channels, dtype, **kwargs
        Additional parameters to pass to StreamBase.

    Attributes
    ----------
    txq : Ringbuffer
        RingBuffer used for writing audio data to the output Portaudio stream.
    rxq : Ringbuffer
        RingBuffer used for reading audio data from the input Portaudio stream.
    framesize : int
        The audio frame size in bytes. Equivalent to channels*samplesize.
    """
    def __init__(self, kind, qsize=PA_QSIZE, blocksize=None, device=None,
                 channels=None, dtype=None, **kwargs):
        # unfortunately we need to figure out the framesize before allocating
        # the stream in order to be able to pass our user_data
        self.txq = self.rxq = None
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
            self.txq = _pa_ringbuffer.RingBuffer(self.framesize[1], qsize)
            self.rxq = _pa_ringbuffer.RingBuffer(self.framesize[0], qsize)
        else:
            parameters, dtype, samplesize, _ = _sd._get_stream_parameters(
                kind, device, channels, dtype, None, None, None)
            self.framesize = samplesize*parameters.channelCount
            if kind == 'output':
                self.txq = _pa_ringbuffer.RingBuffer(self.framesize, qsize)
            if kind == 'input':
                self.rxq = _pa_ringbuffer.RingBuffer(self.framesize, qsize)

        # Set up the C portaudio callback
        self._callback_info = _ffi.NULL
        if kwargs.get('callback', None) is None:
            userdata = {'status': 0,
                        'duplexity': ['input', 'output', 'duplex'].index(kind) + 1,
                        'frame_count': 0,
                        'errorMsg': b'',
                        'lastTime': 0 }
            if self.rxq is not None:
                userdata['rxq'] = _ffi.cast('PaUtilRingBuffer*', self.rxq._ptr)
            if self.txq is not None:
                userdata['txq'] = _ffi.cast('PaUtilRingBuffer*', self.txq._ptr)
                
            self._callback_info = _ffi.new("Py_PsBufferedStream*", userdata)
            kwargs['userdata'] = self._callback_info
            kwargs['callback'] = _ffi.addressof(_libps, 'callback')
            kwargs['wrap_callback'] = None

        super(_BufferedStreamBase, self).__init__(kind, blocksize=blocksize,
                                                  device=device,
                                                  channels=channels,
                                                  dtype=dtype, **kwargs)

        if isinstance(self._device, int):
            self._device_name = _sd.query_devices(self._device)['name']
        else:
            self._device_name = tuple(_sd.query_devices(dev)['name'] for dev in self._device)

    @property
    def status(self):
        return _sd.CallbackFlags(int(self._callback_info.status))

    @property
    def frame_count(self):
        return int(self._callback_info.frame_count)

    @property
    def max_delta(self):
        return float(self._callback_info.max_dt)

    @property
    def min_delta(self):
        return float(self._callback_info.min_dt)

    def _check_callback(self):
        msg = _ffi.string(self._callback_info.errorMsg)
        if len(msg):
            raise AudioBufferError(msg)

    def abort(self):
        super(_BufferedStreamBase, self).abort()
        self._check_callback()

    def stop(self):
        super(_BufferedStreamBase, self).stop()
        self._check_callback()

    def close(self):
        super(_BufferedStreamBase, self).close()
        self._check_callback()

    def __repr__(self):
        return ("{0}({1._device_name!r}, samplerate={1._samplerate:.0f}, "
                "channels={1._channels}, dtype={1._dtype!r}, blocksize={1._blocksize})").format(self.__class__.__name__, self)

class _InputStreamMixin(object):
    def blockstream(self, overlap=0, always_2d=False, copy=False):
        """
        Similar to SoundFile.blocks. Returns an iterator over audio chunks read from
        a Portaudio stream. Can be either half-duplex (recording-only) or
        full-duplex if an input file is supplied.

        Parameters
        ----------
        overlap : int
            Number of frames to overlap across blocks.
        always_2d : bool
            Always returns blocks 2 dimensional arrays. Only valid if you have numpy
            installed.
        copy : bool
            Whether to return copies of blocks. By default a view is returned.

        Yields
        ------
        array
            ndarray or memoryview object with `blocksize` elements.
        """
        blocksize = self.blocksize
        assert not self.active, "Stream has already been started!"
        if not blocksize:
            raise ValueError("Requires a fixed known blocksize")
        if overlap >= blocksize:
            raise ValueError("Overlap must be less than blocksize")
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

        if _np is None:
            outbuff = memoryview(bytearray(blocksize*framesize))
        else:
            outbuff = _np.zeros((blocksize, channels) if always_2d else blocksize*channels, dtype=dtype)

        if copy:
            if _np is None:
                yielder = lambda buff: memoryview(bytearray(buff))
            else:
                yielder = _np.copy
        else:
            yielder = None

        incframes = blocksize - overlap
        dt = incframes / self.samplerate
        ringbuff = self.rxq
        with self:
            while self.active:
                nframes = ringbuff.read(outbuff[overlap:], incframes)
                if nframes == 0:
                    _time.sleep(dt)
                    continue

                yield yielder(outbuff[:overlap + nframes]) if copy else outbuff[:overlap + nframes]

                outbuff[:-incframes] = outbuff[incframes:]

                if nframes < incframes:
                    break

class BufferedInputStream(_InputStreamMixin, _BufferedStreamBase):
    def __init__(self, **kwargs):
        super(BufferedInputStream).__init__(self, 'input', **kwargs)

class BufferedOutputStream(_BufferedStreamBase):
    def __init__(self, **kwargs):
        super(BufferedOutputStream).__init__(self, 'output', **kwargs)

class BufferedStream(BufferedInputStream, BufferedOutputStream):
    def __init__(self, **kwargs):
        _BufferedStreamBase.__init__(self, 'duplex', **kwargs)

class _ThreadedStreamBase(_BufferedStreamBase):
    """
    This class builds on the BufferedStream class by adding the ability to
    register functions for reading (qreader) and writing (qwriter) audio data
    which run in their own threads. However, the qreader and qwriter threads are
    optional; this allows the use of a 'duplex' stream which e.g. has a
    dedicated thread for writing data but for which data is read in the main
    thread.

    An important advantage with this class is that it properly handles any
    exceptions raised in the qreader and qwriter threads. Specifcally, if an
    exception is raised, the stream will be aborted and the exception re-raised
    in the main thread.

    Parameters
    -----------
    blocksize : int
        Portaudio buffer size. If None, the Portaudio backend will automatically
        determine a size.
    qreader : function
        Function that handles reading from the receive queue. Will be called in
        a seperate thread.
    qwriter : function
        Function that handles writing to the transmit queue. Will be called in
        a seperate thread.

    Other Parameters
    -----------------
    kind, **kwargs
        Additional parameters to pass to BufferedStreamBase.

    Attributes
    -----------
    txt : Thread object
        Daemon thread object that handles writing data to the output ring buffer.
    rxt : Thread object
        Daemon thread object that handles reading data from the input ring buffer.
    """
    def __init__(self, kind, qreader=None, qwriter=None, blocksize=None, **kwargs):
        if qreader is None and qwriter is None:
            raise ValueError("No qreader or qwriter function given.")

        super(_ThreadedStreamBase, self).__init__(kind, blocksize=blocksize, **kwargs)

        self._exit = _threading.Event()
        self._exc = _queue.Queue()

        if (kind == 'duplex' or kind == 'output') and qwriter is not None:
            txt = _threading.Thread(target=self._qrwwrapper, args=(self.txq, qwriter))
            txt.daemon = True
            self.txt = txt
        else:
            self.txt = None

        if (kind == 'duplex' or kind == 'input') and qreader is not None:
            rxt = _threading.Thread(target=self._qrwwrapper, args=(self.rxq, qreader))
            rxt.daemon = True
            self.rxt = rxt
        else:
            self.rxt = None

    def _raise_exceptions(self):
        if self._exc.empty():
            return

        exc = self._exc.get()
        if isinstance(exc, tuple):
            exctype, excval, exctb = exc
            if exctype is not None:
                excval = exctype(excval)
            try:
                raise excval.with_traceback(exctb)
            except AttributeError:
                exec("raise excval, None, exctb")

        raise exc

    def _set_exception(self, exc=None):
        # ignore subsequent exceptions
        if not self._exc.empty():
            return
        if exc is None:
            exc = _sys.exc_info()
        self._exc.put(exc)

    def _qrwwrapper(self, queue, qrwfunc):
        """
        Wrapper function for the qreader and qwriter threads which acts as a
        kind of context manager.
        """
        try:
            qrwfunc(self, queue)
        except:
            # Raise the exception in the main thread
            self._set_exception()

            # suppress the exception in this child thread
            try:    self.abort()
            except: pass

    def _stopiothreads(self):
        currthread = _threading.current_thread()

        self._exit.set()
        if self.rxt is not None and self.rxt.is_alive() and self.rxt != currthread:
            self.rxt.join()

        if self.txt is not None and self.txt.is_alive() and self.txt != currthread:
            self.txt.join()

    def start(self):
        self._exit.clear()
        if self.txt is not None:
            self.txt.start()
            while self.txq.write_available and self.txt.is_alive():
                _time.sleep(0.001)
        if self.rxt is not None:
            self.rxt.start()
        super(_ThreadedStreamBase, self).start()

    def abort(self):
        super(_ThreadedStreamBase, self).abort()
        self._stopiothreads()

    def stop(self):
        super(_ThreadedStreamBase, self).stop()
        self._stopiothreads()

    def close(self):
        super(_ThreadedStreamBase, self).close()
        self._stopiothreads()
        self._raise_exceptions()

class ThreadedInputStream(_InputStreamMixin, _ThreadedStreamBase):
    def __init__(self, **kwargs):
        super(ThreadedInputStream).__init__(self, 'input', **kwargs)

class ThreadedOutputStream(_ThreadedStreamBase):
    def __init__(self, **kwargs):
        super(ThreadedOutputStream).__init__(self, 'output', **kwargs)

class ThreadedStream(ThreadedInputStream, ThreadedOutputStream):
    def __init__(self, **kwargs):
        _ThreadedStreamBase.__init__(self, 'duplex', **kwargs)

class _SoundFileStreamBase(_ThreadedStreamBase):
    """
    This helper class basically gives you two things:

        1) it provides complete qreader and qwriter functions for SoundFile
           objects (or anything that can be opened as a SoundFile object)

        2) it automatically sets parameters for the stream based on the input
           file and automatically sets parameters for the output file based on
           the output stream.

    Parameters
    -----------
    inpf : SoundFile compatible input
        Input file to stream to audio device. The input file will determine the
        samplerate and number of channels for the audio stream.
    outf : SoundFile compatible input
        Output file to capture data from audio device. If a SoundFile is not
        passed, the output file parameters will be determined from the output
        audio stream.
    fileblocksize : int
        How many frames to read/write to the queue at a time. Ideally, this
        should be greater than or equal to the average Portaudio buffer size in
        order not to cause underflow/overflow. Setting this parameter does not
        effect the Portaudio buffer size.

    Attributes
    ------------
    inp_fh : SoundFile
        The file object to write to the output ring buffer.
    out_fh : SoundFile
        The file object to capture data from the input ring buffer.

    Other Parameters
    ----------------------
    qreader, qwriter, kind, fileblocksize, blocksize, **kwargs
        Additional parameters to pass to ThreadedStreamBase.
    """
    def __init__(self, kind, inpf=None, outf=None, qreader=None, qwriter=None, sfkwargs={}, fileblocksize=None, blocksize=None, **kwargs):
        # We're playing an audio file, so we can safely assume there's no need
        # to clip
        if kwargs.get('clip_off', None) is None:
            kwargs['clip_off'] = True

        if not (fileblocksize or blocksize):
            raise ValueError("One or both of fileblocksize and blocksize must be non-zero.")

        # At this point we don't care what 'kind' the stream is, only whether
        # the input/output is None which determines whether qreader/qwriter
        # functions should be registered
        self._inpf = inpf
        if inpf is not None:
            inp_fh = inpf
            if not isinstance(inpf, _sf.SoundFile):
                inp_fh = _sf.SoundFile(inpf)
            if kwargs.get('samplerate', None) is None:
                kwargs['samplerate'] = inp_fh.samplerate
            if kwargs.get('channels', None) is None:
                kwargs['channels'] = inp_fh.channels
            if qwriter is None:
                qwriter = self._soundfilereader
        else:
            inp_fh = inpf

        # We need to set the qreader here; output file parameters will known
        # once we open the stream
        self._outf = outf
        if outf is not None and qreader is None:
            qreader = self._soundfilewriter

        super(_SoundFileStreamBase, self).__init__(kind, qreader=qreader,
                                                   qwriter=qwriter,
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
        if outf is not None and not isinstance(outf, _sf.SoundFile):
            if inp_fh is not None:
                if sfkwargs.get('endian', None) is None:
                    sfkwargs['endian'] = inp_fh.endian
                if (sfkwargs.get('format', outext) == inp_fh.format.lower()
                    and sfkwargs.get('subtype', None) is None):
                    sfkwargs['subtype'] = inp_fh.subtype
            if sfkwargs.get('channels', None) is None:
                sfkwargs['channels'] = self.channels[0] if kind == 'duplex' else self.channels
            if sfkwargs.get('samplerate', None) is None:
                sfkwargs['samplerate'] = int(self.samplerate)
            if sfkwargs.get('mode', None) is None:
                sfkwargs['mode'] = 'w+b'
            out_fh = _sf.SoundFile(outf, **sfkwargs)
        else:
            out_fh = outf

        self.inp_fh = inp_fh
        self.out_fh = out_fh
        self.fileblocksize = fileblocksize or self.blocksize

    # Default handler for writing input from a ThreadedStream to a SoundFile object
    @staticmethod
    def _soundfilewriter(stream, ringbuff):
        try:
            framesize = stream.framesize[0]
            dtype = stream.dtype[0]
        except TypeError:
            framesize = stream.framesize
            dtype = stream.dtype

        dt = stream.fileblocksize / stream.samplerate
        buff = memoryview(bytearray(stream.fileblocksize*framesize))
        while not stream._exit.is_set():
            nframes = ringbuff.read(buff)
            if nframes == 0:
                _time.sleep(dt)
                continue
            stream.out_fh.buffer_write(buff[:nframes*framesize], dtype=dtype)

    # Default handler for reading input from a SoundFile object and writing it to a
    # ThreadedStream
    @staticmethod
    def _soundfilereader(stream, ringbuff):
        try:
            framesize = stream.framesize[1]
            dtype = stream.dtype[1]
        except TypeError:
            framesize = stream.framesize
            dtype = stream.dtype

        dt = stream.fileblocksize / stream.samplerate
        buff = memoryview(bytearray(stream.fileblocksize*framesize))
        while not stream._exit.is_set():
            nframes = min(ringbuff.write_available, stream.fileblocksize)
            if nframes == 0:
                _time.sleep(dt)
                continue

            readframes = stream.inp_fh.buffer_read_into(buff[:nframes*framesize], dtype=dtype)
            ringbuff.write(buff, readframes)
            if readframes < nframes:
                break

    def close(self):
        try:
            super(_SoundFileStreamBase, self).close()
        finally:
            if isinstance(self._outf, _basestring):
                self.out_fh.close()
            if isinstance(self._inpf, _basestring):
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
    def __init__(self, outf, sfkwargs={}, fileblocksize=None, **kwargs):
        super(SoundFileInputStream, self).__init__('input', outf=outf,
                                                   sfkwargs=sfkwargs,
                                                   fileblocksize=fileblocksize,
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
    def __init__(self, inpf, qsize=PA_QSIZE, fileblocksize=None, **kwargs):
        super(SoundFileOutputStream, self).__init__('output', inpf=inpf,
                                                    qsize=qsize,
                                                    fileblocksize=fileblocksize,
                                                    **kwargs)

class SoundFileStream(SoundFileInputStream, SoundFileOutputStream):
    """
    Full duplex audio file streamer. Note that only one of inpf and outf is
    required. This allows you to e.g. use a SoundFile as input but implement
    your own qreader and/or read from the queue in the main thread.
    """
    def __init__(self, inpf=None, outf=None, qsize=PA_QSIZE, sfkwargs={}, fileblocksize=None, **kwargs):
        # If you're not using soundfiles for the input or the output, then you
        # should probably be using the Buffered or ThreadedStream class
        if inpf is None and outf is None:
            raise ValueError("No input or output file given.")

        _SoundFileStreamBase.__init__(self, 'duplex', inpf=inpf, outf=outf, qsize=qsize,
                                      fileblocksize=fileblocksize,
                                      sfkwargs=sfkwargs, **kwargs)

def blockstream(inpf=None, blocksize=512, overlap=0, always_2d=False, copy=False, qwriter=None, streamclass=None, **kwargs):
    """
    Read audio data in chunks from a Portaudio stream. Can be either half-duplex
    (recording-only) or full-duplex (if an input file is supplied).

    Parameters
    ------------
    inpf : SoundFile compatible input or None
        Optional input stimuli.
    blocksize : int
        Portaudio stream buffer size. Must be non-zero.
    overlap : int
        Number of frames to overlap across blocks.
    always_2d : bool
        Always returns blocks 2 dimensional arrays. Only valid if you have numpy
        installed.
    copy : bool
        Whether to return copies of blocks. By default a view is returned.

    Other Parameters
    -----------------
    qwriter : function
        Function that handles writing to the audio transmit queue. Can be used
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
            streamclass = BufferedInputStream if qwriter is None else ThreadedStream
        stream = streamclass(blocksize=blocksize, qwriter=qwriter, **kwargs)
    else:
        if streamclass is None:
            streamclass = SoundFileStream
        stream = streamclass(inpf, blocksize=blocksize, qwriter=qwriter, **kwargs)

    return stream.blockstream(overlap, copy, always_2d)

# Used just for the pastream app
def SoundFileStreamFactory(inpf=None, outf=None, **kwargs):
    if inpf is not None and outf is not None:
        Streamer = SoundFileStream
        ioargs = (inpf, outf)
    elif inpf is not None:
        Streamer = SoundFileOutputStream
        ioargs = (inpf,)
        kwargs.pop('sfkwargs', None)
        kwargs.pop('qreader', None)
    elif outf is not None:
        Streamer = SoundFileInputStream
        ioargs = (outf,)
        kwargs.pop('qwriter', None)
    else:
        raise SystemExit("No input or output selected.")

    return Streamer(*ioargs, **kwargs)

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
        assert intarg > 0, "Queue size must be a positive value."
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

    parser.add_argument("-q", "--qsize", type=posint, default=PA_QSIZE,
help="File transmit buffer size (in units of frames). Increase for smaller blocksizes.")

    parser.add_argument("-B", "--file-blocksize", type=int,
                         help='''\
Only used for special cases where you want to set the file block size
differently than the portaudio buffer size. You most likely don't need to use
this option. (In units of frames).''')

    devopts = parser.add_argument_group("I/O device stream options")

    devopts.add_argument("-d", "--device", type=devtype,
                         help='''\
Audio device name expression or index number. Defaults to the PortAudio default device.''')

    devopts.add_argument("-b", "--blocksize", type=int,
                         help="PortAudio buffer size and file block size (in units of frames).")

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
    stream = SoundFileStreamFactory(args.input, args.output, qsize=args.qsize,
                                    sfkwargs={
                                        'endian': args.endian,
                                        'subtype': args.encoding,
                                        'format': args.file_type
                                    },
                                    device=args.device,
                                    channels=args.channels,
                                    dtype=args.dtype,
                                    fileblocksize=args.file_blocksize,
                                    samplerate=args.samplerate,
                                    blocksize=args.blocksize)

    statline = "\r{:8.3f}s {:10d} frames processed, {:>10s} frames available for writing, {:>10s} frames queued for reading\r"
    print("<-", stream.inp_fh if stream.inp_fh is not None else 'null')
    print("--", stream)
    print("->", stream.out_fh if stream.out_fh is not None else 'null')

    nullinp = nullout = None
    if stream.inp_fh is None:
        nullinp = 'n/a'
    if stream.out_fh is None:
        nullout = 'n/a'

    try:
        with stream:
            t1 = _time.time()
            while stream.active:
                _time.sleep(0.1)
                line = statline.format(_time.time()-t1, stream.frame_count,
                                       nullinp or str(stream.txq.write_available),
                                       nullout or str(stream.rxq.read_available))
                _sys.stdout.write(line); _sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        print()

    print("Delta range (ms): [ {:7.3f}, {:7.3f}]".format(1e3*(stream.min_delta), 1e3*(stream.max_delta)))
    print("Nominal: {:7.3f}".format(1e3*stream.fileblocksize/stream.samplerate))

    return 0

if __name__ == '__main__':
    _sys.exit(_main())
