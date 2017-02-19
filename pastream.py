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
from __future__ import print_function
try:
    import Queue as queue
except ImportError:
    import queue
try:
    basestring
except NameError:
    basestring = (str, bytes)
import threading
import time
import sys
import sounddevice as sd
import soundfile as sf
import traceback
import rtmixer

__usage__ = "%(prog)s [options] [-d device] input output"


TXQSIZE = 1<<16 # Number of frames to buffer for transmission


class AudioBufferError(Exception):
    pass

class QueuedStreamBase(sd._StreamBase):
    """
    This class adds a python Queue for reading and writing audio data. This
    double buffers the audio data so that any processing is kept out of the time
    sensitive audio callback function. For maximum flexibility, receive queue
    data is a bytearray object; transmit queue data should be of a buffer type
    where each element is a single byte.

    Notes:
    
    Using a queue requires that we use a fixed, predetermined value for the
    portaudio blocksize.

    If the receive buffer fills or the transmit buffer is found empty during a
    callback the audio stream is aborted and an exception is raised.

    During recording, the end of the stream is signaled by a 'None' value placed
    into the receive queue.

    During playback, the end of the stream is signaled by an item on the queue
    that is smaller than blocksize*channels*samplesize bytes.

    """
    def __init__(self, blocksize=None, qsize=TXQSIZE, kind='duplex', **kwargs):
        if kwargs.get('callback', None) is None: 
            if kind == 'input':
                kwargs['callback'] = self.icallback
            elif kind == 'output':
                kwargs['callback'] = self.ocallback
            else:
                kwargs['callback'] = self.iocallback

        kwargs['wrap_callback'] = 'buffer'
        super(QueuedStreamBase, self).__init__(kind=kind, blocksize=blocksize, **kwargs)

        self.status = sd.CallbackFlags()
        self.frame_count = 0
        self._closed = False

        if isinstance(self._device, int):
            self._devname = sd.query_devices(self._device)['name']
        else:
            self._devname = tuple(sd.query_devices(dev)['name'] for dev in self._device)

        if kind == 'duplex':
            self.framesize = self.samplesize[0]*self.channels[0], self.samplesize[1]*self.channels[1]
        else:
            self.framesize = self.samplesize*self.channels

        if kind == 'duplex' or kind == 'output':
            self.txq = rtmixer.RingBuffer(self.framesize[1] if kind=='duplex' else self.framesize, qsize)
            self.txq.event = threading.Event()
            self.txq.event.set()
        else:
            self.txq = None

        if kind == 'duplex' or kind == 'input':
            self.rxq = rtmixer.RingBuffer(self.framesize[0] if kind=='duplex' else self.framesize, qsize)
            self.rxq.event = threading.Semaphore(0)
        else:
            self.rxq = None

    def icallback(self, in_data, frame_count, time_info, status):
        self.status |= status
        if status._flags&0xF:
            self._set_exception(AudioBufferError(str(status)))
            raise sd.CallbackAbort

        if self.rxq.write_available < frame_count:
            self._set_exception(AudioBufferError("Receive queue is full."))
            raise sd.CallbackAbort

        iframes = self.rxq.write(in_data, frame_count)
        self.rxq.event.release()

        self.frame_count += frame_count

    def ocallback(self, out_data, frame_count, time_info, status):
        self.status |= status
        if status._flags&0xF:
            self._set_exception(AudioBufferError(str(status)))
            raise sd.CallbackAbort

        oframes = self.txq.read(out_data, frame_count)
        self.txq.event.set()

        # This is our last callback!
        if oframes < frame_count:
            self.frame_count += oframes
            raise sd.CallbackStop

        self.frame_count += frame_count

    def iocallback(self, in_data, out_data, frame_count, time_info, status):
        self.status |= status
        if status._flags&0xF:
            self._set_exception(AudioBufferError(str(status)))
            raise sd.CallbackAbort

        if self.rxq.write_available < frame_count:
            self._set_exception(AudioBufferError("Receive queue is full."))
            raise sd.CallbackAbort

        iframes = self.rxq.write(in_data, frame_count)
        self.rxq.event.release()

        oframes = self.txq.read(out_data, frame_count)
        self.txq.event.set()

        # This is our last callback!
        if oframes < frame_count:
            self.frame_count += oframes
            self.rxq.event.release()
            raise sd.CallbackStop

        self.frame_count += frame_count

    def _closequeues(self):
        if not self._closed:
            self._closed = True
        else:
            return

        if self.rxq is not None:
            self.rxq.flush()
            self.rxq.event.release()

        if self.txq is not None:
            self.txq.flush()
            self.txq.event.set()

    def abort(self):
        super(QueuedStreamBase, self).abort()
        self._closequeues()

    def start(self):
        if self.rxq is not None:
            while self.rxq.event.acquire(0): pass
        if self.txq is not None:
            if not self.txq.write_available:
                self.txq.event.clear()
            else:
                self.txq.event.set()
        super(QueuedStreamBase, self).start()

    def stop(self):
        super(QueuedStreamBase, self).stop()
        self._closequeues()

    def close(self):
        super(QueuedStreamBase, self).close()
        self._closequeues()

    def __repr__(self):
        return ("{0}({1._devname!r}, samplerate={1._samplerate:.0f}, "
                "channels={1._channels}, dtype={1._dtype!r}, blocksize={1._blocksize})").format(self.__class__.__name__, self)

class ThreadedStreamBase(QueuedStreamBase):
    """
    This class builds on the QueuedStream class by adding the ability to
    register functions for reading (qreader) and writing (qwriter) audio data
    which run in their own threads. However, the qreader and qwriter threads are
    optional; this allows the use of a 'duplex' stream which e.g. has a
    dedicated thread for writing data but for which data is read in the main
    thread.

    An important advantage with this class is that it properly handles any
    exceptions raised in the qreader and qwriter threads. Specifcally, if an
    exception is raised, the stream will be aborted and the exception re-raised
    in the main thread.
    """
    def __init__(self, blocksize=None, qreader=None, qwriter=None, kind='duplex', **kwargs):
        if qreader is None and qwriter is None: 
            raise ValueError("No qreader or qwriter function given.")

        super(ThreadedStreamBase, self).__init__(kind=kind, blocksize=blocksize, **kwargs)

        self._exit = threading.Event()
        self._exc = queue.Queue()

        if (kind == 'duplex' or kind == 'output') and qwriter is not None:
            txt = threading.Thread(target=self._qrwwrapper, args=(self.txq, qwriter))
            txt.daemon = True
            self.txt = txt
        else:
            self.txt = None

        if (kind == 'duplex' or kind == 'input') and qreader is not None:
            rxt = threading.Thread(target=self._qrwwrapper, args=(self.rxq, qreader))
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
            try:
                raise exctype(excval).with_traceback(exctb)
            except AttributeError:
                exec("raise excval, None, exctb")

        raise exc

    def _set_exception(self, exc=None):
        # ignore subsequent exceptions
        if not self._exc.empty():
            return
        if exc is None:
            exc = sys.exc_info()
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
        currthread = threading.current_thread()

        self._exit.set()
        if self.rxt is not None and self.rxt.is_alive() and self.rxt != currthread:
            self.rxt.join()

        if self.txt is not None and self.txt.is_alive() and self.txt != currthread:
            self.txt.join()

    def start(self):
        self._exit.clear()
        if self.txt is not None:
            self.txq.event.set()
            self.txt.start()
            while self.txq.write_available and self.txt.is_alive(): 
                time.sleep(0.001)

        if self.rxt is not None:
            self.rxt.start()
        super(ThreadedStreamBase, self).start()

    def abort(self):
        super(ThreadedStreamBase, self).abort()
        self._stopiothreads()

    def stop(self):
        super(ThreadedStreamBase, self).stop()
        self._stopiothreads()

    def close(self):
        super(ThreadedStreamBase, self).close()
        self._stopiothreads()
        self._raise_exceptions()

def _soundfilewriter(stream, ringbuff):
    try:               
        framesize = stream.framesize[0]
        dtype = stream.dtype[0]
    except TypeError: 
        framesize = stream.framesize
        dtype = stream.dtype    

    buff = memoryview(bytearray(stream.fileblocksize*framesize))
    while True:
        ringbuff.event.acquire()

        nframes = ringbuff.read(buff)
        if nframes == 0:
            break
        stream.out_fh.buffer_write(buff[:nframes*framesize], dtype=dtype)

def _soundfilereader(stream, ringbuff):
    try:               
        framesize = stream.framesize[1]
        dtype = stream.dtype[1]
    except TypeError: 
        framesize = stream.framesize
        dtype = stream.dtype    

    buff = memoryview(bytearray(stream.fileblocksize*framesize))
    while not stream._exit.is_set():
        if not ringbuff.event.wait(timeout=1):
            raise AudioBufferError("time out waiting to write data")

        nframes = min(ringbuff.write_available, stream.fileblocksize)
        if nframes == 0: 
            ringbuff.event.clear()
            continue

        readframes = stream.inp_fh.buffer_read_into(buff[:nframes*framesize], dtype=dtype)
        ringbuff.write(buff, readframes)
        if readframes < nframes:
            break

class SoundFileStreamBase(ThreadedStreamBase):
    """
    This helper class basically gives you two things: 

        1) it provides complete qreader and qwriter functions for SoundFile
           objects (or anything that can be opened as a SoundFile object)

        2) it automatically sets parameters for the stream based on the input
           file and automatically sets parameters for the output file based on
           the output stream.
    """
    def __init__(self, inpf=None, outf=None, qreader=None, qwriter=None, kind='duplex', sfkwargs={}, fileblocksize=None, blocksize=None, **kwargs):
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
            if not isinstance(inpf, sf.SoundFile):
                inp_fh = sf.SoundFile(inpf)
            if kwargs.get('samplerate', None) is None: 
                kwargs['samplerate'] = inp_fh.samplerate
            if kwargs.get('channels', None) is None: 
                kwargs['channels'] = inp_fh.channels
            if qwriter is None:
                qwriter = _soundfilereader
        else:
            inp_fh = inpf

        # We need to set the qreader here; output file parameters will known
        # once we open the stream
        self._outf = outf
        if outf is not None and qreader is None:
            qreader = _soundfilewriter                
            
        super(SoundFileStreamBase, self).__init__(qreader=qreader,
                                                  qwriter=qwriter, kind=kind,
                                                  blocksize=blocksize, **kwargs)

        # Try and determine the file extension here; we need to know it if we
        # want to try and set a default subtype for the output
        try:
            outext = getattr(outf, 'name', outf).rsplit('.', 1)[1].lower()
        except AttributeError:
            outext = None

        # If the output file hasn't already been opened, we open it here using
        # the input file and output stream settings, plus any user supplied
        # arguments
        if outf is not None and not isinstance(outf, sf.SoundFile):
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
            out_fh = sf.SoundFile(outf, **sfkwargs)
        else:
            out_fh = outf

        self.inp_fh = inp_fh
        self.out_fh = out_fh
        self.fileblocksize = fileblocksize or self.blocksize

    def close(self):
        try:
            super(SoundFileStreamBase, self).close()
        finally:
            if self._outf != self.out_fh:
                self.out_fh.close()
            if self._inpf != self.inp_fh:
                self.inp_fh.close()

class SoundFileInputStream(SoundFileStreamBase):
    def __init__(self, outf, sfkwargs={}, fileblocksize=None, **kwargs):
        super(SoundFileInputStream, self).__init__(outf=outf, kind='input',
                                                   sfkwargs=sfkwargs,
                                                   fileblocksize=fileblocksize,
                                                   **kwargs)

class SoundFileOutputStream(SoundFileStreamBase):
    def __init__(self, inpf, qsize=TXQSIZE, fileblocksize=None, **kwargs):
        super(SoundFileOutputStream, self).__init__(inpf=inpf, qsize=qsize,
                                                    fileblocksize=fileblocksize,
                                                    kind='output', **kwargs)

class SoundFileStream(SoundFileStreamBase):
    """
    Note that only one of inpf and outf is required. This allows you to e.g. use
    a SoundFile as input but implement your own qreader and/or read from the
    queue in the main thread.
    """
    def __init__(self, inpf=None, outf=None, qsize=TXQSIZE, sfkwargs={}, fileblocksize=None, **kwargs):
        # If you're not using soundfiles for the input or the output, then you
        # should probably be using the Queued or ThreadedStream class
        if inpf is None and outf is None: 
            raise ValueError("No input or output file given.")

        super(SoundFileStream, self).__init__(inpf=inpf, outf=outf, qsize=qsize,
                                              fileblocksize=fileblocksize, sfkwargs=sfkwargs,
                                              kind='duplex', **kwargs)

def blockstream(inpf=None, blocksize=512, overlap=0, always_2d=False, copy=False, qwriter=None, streamclass=None, **kwargs):
    import numpy as np
    assert blocksize is not None and blocksize > 0, "Requires a fixed known blocksize"

    incframes = blocksize-overlap
    if inpf is None and qwriter is None:
        if streamclass is None: 
            streamclass = QueuedStreamBase
            kwargs['kind'] = 'input'
        stream = streamclass(blocksize=incframes, **kwargs)
        dtype = stream.dtype
        channels = stream.channels
    else:
        if streamclass is None: 
            streamclass = SoundFileStream
        stream = streamclass(inpf, blocksize=incframes, **kwargs)
        dtype = stream.dtype[0]
        channels = stream.channels[0]

    if channels > 1:
        always_2d = True

    ringbuff = stream.rxq
    outbuff = np.zeros((blocksize, channels) if always_2d else blocksize*channels, dtype=dtype)
    with stream:
        while stream.active:
            ringbuff.event.acquire()

            nframes = ringbuff.read(outbuff[overlap:])
            if nframes:
                yield outbuff[:overlap+nframes].copy() if copy else outbuff[:overlap+nframes]

            if nframes < blocksize:
                break

            outbuff[:-incframes] = outbuff[incframes:]

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

def get_parser(parser=None):
    from argparse import Action, ArgumentParser, RawDescriptionHelpFormatter

    if parser is None:
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                                fromfile_prefix_chars='@',
                                usage=__usage__,
                                description=__doc__)
        parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()

    class ListStreamsAction(Action):
        def __call__(*args, **kwargs):
            print(sd.query_devices())
            sys.exit(0)

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

    parser.add_argument("-q", "--qsize", type=posint, default=TXQSIZE,
help="File transmit buffer size (in units of frames). Increase for smaller blocksizes.")

    devopts = parser.add_argument_group("I/O device stream options")

    devopts.add_argument("-d", "--device", type=devtype,
                         help='''\
Audio device name expression or index number. Defaults to the PortAudio default device.''')

    devopts.add_argument("-b", "--blocksize", type=int,
                         help="PortAudio buffer size and file block size (in units of frames).")

    devopts.add_argument("-B", "--file-blocksize", type=int,
                         help='''\
Only used for special cases where you want to set the file block size
differently than the portaudio buffer size. You most likely don't need to use
this option. (In units of frames).''')

    devopts.add_argument("-f", "--format", dest='dtype',
                         choices=sd._sampleformats.keys(),
                         help='''Sample format of device I/O stream.''')

    devopts.add_argument("-c", "--channels", type=int,
                         help="Number of channels.")

    devopts.add_argument("-r", "--rate", dest='samplerate',
                         type=lambda x: int(float(x[:-1])*1000) if x.endswith('k') else int(x),
                         help="Sample rate in Hz. Add a 'k' suffix to specify kHz.")

    fileopts = parser.add_argument_group('''\
Output file format options''')

    fileopts.add_argument("-t", dest="file_type", choices=sf.available_formats().keys(),
                          type=str.upper,
                          help='''\
Output file type. Typically this is determined from the file extension, but it
can be manually overridden here.''')

    fileopts.add_argument("-e", "--encoding", choices=sf.available_subtypes(),
                          type=str.upper,
                          help="Sample format encoding.")

    fileopts.add_argument("--endian", choices=['file', 'big', 'little'], 
                          help="Byte endianness.")

    return parser

def main(argv=None):
    if argv is None: argv=sys.argv[1:]
    parser = get_parser()
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
            t1 = time.time()
            while stream.active:
                time.sleep(0.1)
                line = statline.format(time.time()-t1, stream.frame_count, 
                                       nullinp or str(stream.txq.write_available),
                                       nullout or str(stream.rxq.read_available))
                sys.stdout.write(line); sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
