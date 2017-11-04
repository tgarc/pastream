"""
Loopback tests for pastream.
"""
from __future__ import print_function
import os, platform, subprocess, io, sys
import pytest
import time
import tempfile
import numpy as np
import numpy.testing as npt
import soundfile as sf
import sounddevice as sd
import pastream as ps


PREAMBLE = 0x7FFFFFFF # Value used for the preamble sequence (before appropriate shifting for dtype)
PREAMBLESIZE = 4096

# Set up the platform specific device
if platform.system() == 'Windows':
    DEVICE_KWARGS = { 'device': 'ASIO4ALL v2, ASIO', 'dtype': 'int32' }
elif platform.system() == 'Darwin':
    raise Exception("Currently no test support for OSX")
else:
    # This is assuming you're using the ALSA device set up by .asoundrc config
    # file in the root of the repository
    DEVICE_KWARGS = { 'device': 'aloop_duplex', 'dtype': 'int32' }

defaults = sd.query_devices(DEVICE_KWARGS['device'])
channels = min(defaults['max_input_channels'], defaults['max_output_channels'])
DEVICE_KWARGS.setdefault('channels', channels)
DEVICE_KWARGS.setdefault('samplerate', defaults['default_samplerate'])

if 'SOUNDDEVICE_DEVICE_NAME' in os.environ:
    DEVICE_KWARGS['device'] = os.environ['SOUNDDEVICE_DEVICE_NAME']
if 'SOUNDDEVICE_DEVICE_BLOCKSIZE' in os.environ:
    DEVICE_KWARGS['blocksize'] = int(os.environ['SOUNDDEVICE_DEVICE_BLOCKSIZE'])
if 'SOUNDDEVICE_DEVICE_DTYPE' in os.environ:
    DEVICE_KWARGS['dtype'] = os.environ['SOUNDDEVICE_DEVICE_DTYPE']
if 'SOUNDDEVICE_DEVICE_CHANNELS' in os.environ:
    DEVICE_KWARGS['channels'] = int(os.environ['SOUNDDEVICE_DEVICE_CHANNELS'])
if 'SOUNDDEVICE_DEVICE_SAMPLERATE' in os.environ:
    DEVICE_KWARGS['samplerate'] = int(os.environ['SOUNDDEVICE_DEVICE_SAMPLERATE'])

_dtype2elementsize = dict(int32=4,int24=3,int16=2,int8=1)
tohex = lambda x: np.vectorize(('{:#0%dx}'%(x.dtype.itemsize*2+2)).format)(x.view('u%d'%x.dtype.itemsize))

def find_soundfile_delay(xf, preamble, dtype):
    pos = xf.tell()
    off = -1

    blocksize = 2048
    unsigned_dtype = dtype if dtype.startswith('u') else 'u'+dtype
    for i,inpblk in enumerate(xf.blocks(blocksize, dtype=dtype, always_2d=True)):
        nonzeros = np.where(inpblk[:, 0].view(unsigned_dtype) == preamble)
        if nonzeros[0].size:
            off = i*blocksize + nonzeros[0][0]
            break
    xf.seek(pos)

    return off

def assert_soundfiles_equal(inp_fh, out_fh, preamble, dtype):
    delay = find_soundfile_delay(out_fh, preamble, dtype)
    assert delay != -1, "Test Preamble pattern not found"
    out_fh.seek(delay)

    mframes = 0
    blocksize = 2048
    unsigned_dtype = 'u' + dtype.lstrip('u')
    inpblk = np.zeros((blocksize, inp_fh.channels), dtype=dtype)
    for outblk in out_fh.blocks(blocksize, dtype=dtype, always_2d=True):
        readframes = inp_fh.buffer_read_into(inpblk[:len(outblk)], dtype=dtype)

        inp = inpblk[:readframes].view(unsigned_dtype)
        out = outblk.view(unsigned_dtype)

        npt.assert_array_equal(inp, out, "Loopback data mismatch")
        mframes += readframes

    print("Matched %d of %d frames; Initial delay of %d frames; %d frames truncated"
          % (mframes, len(inp_fh), delay, len(inp_fh) - inp_fh.tell()))

def randstream(preamblesize, channels, samplesize, dtype='int32', state=None):
    """
    Generates a uniformly random integer signal ranging between the minimum
    and maximum possible values as defined by `samplesize`. The random signal
    is preceded by a constant level equal to the maximum positive integer
    value for `preamblesize` samples which can be used in testing to find the
    beginning of a recording.

    """    
    shift = 8 * (4 - samplesize)
    minval = -(PREAMBLE + 1 >> shift)
    maxval =   PREAMBLE     >> shift

    preamble = np.zeros((preamblesize, channels), dtype=dtype)
    preamble[:] = maxval << shift

    if state is None:
        randint = np.random.randint
    else:
        randint = state.randint

    n = (yield preamble)
    while True:
        block = randint(minval, maxval+1, (n, channels)) << shift
        n = (yield block.astype(dtype))

def looper(stream, maxframes, **kwargs):
    channels = stream.channels
    samplesize = stream.samplesize
    dtype = stream.dtype

    def writer(stream, ringbuff, rblockgen, loop=False):
        ringbuff.write(next(rblockgen))
        while not stream.finished:
            frames = ringbuff.write_available
            if not frames:
                time.sleep(stream.latency[1])
                continue
            ringbuff.write(rblockgen.send(frames))
            time.sleep(stream.latency[1])

    # Generate a seed from which we can generate a common random stream
    seed = np.random.randint(0, (1 << 31) - 1, dtype='int')
    state1 = np.random.RandomState(seed)
    state2 = np.random.RandomState(seed)

    # Set output random stream generator (this is our 'playback')
    rblockgen = randstream(PREAMBLESIZE, channels[1], samplesize[1], dtype[1], state1)
    stream.set_source(writer, args=(rblockgen,))

    # Grab another generator that will give the same random sequence
    rblockgen = randstream(PREAMBLESIZE, channels[0], samplesize[0], dtype[0], state2)

    preamblebuff = ps.RingBuffer(samplesize[0] * channels[0], buffer=next(rblockgen))
    preamblebuff.advance_write_index(PREAMBLESIZE)

    chunkgen = stream.chunks(playback=True, **kwargs)
    frombuffer = lambda x: np.frombuffer(x, dtype=dtype[0]).reshape(-1, channels[0])

    # Read until we find the start of the preamble (this accounts for
    # any possible initial delay in receiving the loopback
    offset = -1
    shift = 8 * (4 - samplesize[0])
    for outframes in chunkgen:
        outframes = frombuffer(outframes)
        matches = outframes == (PREAMBLE >> shift) << shift
        if np.any(matches):
            mindices = np.where(matches)[0]
            offset = mindices[0]
            outframes = outframes[offset:]
            break

    assert offset >= 0, "Preamble not found"

    # Read until we finish receiving the preamble
    readframes = 0
    while True:
        inframes = preamblebuff.read(len(outframes))
        inp = np.frombuffer(inframes, dtype=dtype[0]).reshape(-1, channels[0])
        out = outframes[:len(inp)]
        npt.assert_array_equal(inp, out, "Loopback data mismatch")
        readframes += len(inp)
        if not preamblebuff.read_available:
            outframes = outframes[len(inp):]
            break
        outframes = frombuffer(next(chunkgen))
    mframes = readframes

    # Loop forever checking we match the psuedorandom signal
    while mframes < maxframes:
        inframes = rblockgen.send(len(outframes))
        npt.assert_array_equal(inframes, outframes, "Loopback data mismatch")
        mframes += len(outframes)
        outframes = frombuffer(next(chunkgen))

@pytest.fixture
def random_soundfile_input(tmpdir, scope='session'):
    elementsize = _dtype2elementsize[DEVICE_KWARGS['dtype']]
    dtype = DEVICE_KWARGS['dtype']
    samplerate = int(DEVICE_KWARGS['samplerate'])

    # we don't use an actual TemporaryFile because they don't support multiple
    # file handles on windows
    rdmf= open(tempfile.mktemp(dir=str(tmpdir)), 'w+b')
    rdm_fh = sf.SoundFile(rdmf, 'w+',
                          samplerate,
                          DEVICE_KWARGS['channels'],
                          'PCM_'+['8','16','24','32'][elementsize-1],
                          format='wav')

    with rdm_fh:
        blocks = randstream(PREAMBLESIZE, rdm_fh.channels, elementsize, dtype)
        rdm_fh.write(next(blocks))
        rdm_fh.write(blocks.send(samplerate))
        rdm_fh.seek(0)
        
        if DEVICE_KWARGS['dtype'] == 'int24':
            # Tell the OS it's a 32-bit stream and ignore the extra zeros
            # because 24 bit streams are annoying to deal with
            dtype = 'int32'

        shift = 8*(4-elementsize)

        yield rdm_fh, (PREAMBLE>>shift)<<shift, dtype

@pytest.fixture
def devargs():
    return dict(DEVICE_KWARGS)

def test_chunks_ndarray_loopback(devargs):
    with ps.DuplexStream(**devargs) as stream:
        looper(stream, stream.samplerate)

def test_chunks_bytearray_loopback(devargs):
    with ps.DuplexStream(**devargs) as stream:
        buffer = bytearray(stream.samplesize[0]*stream.channels[0]*ps._PA_BUFFERSIZE)
        looper(stream, stream.samplerate, out=buffer)

def test_soundfilestream_loopback(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    devargs['dtype'] = dtype

    outf = tempfile.TemporaryFile()
    with ps.DuplexStream(**devargs) as stream:
        out_fh = stream.to_file(outf, 'w+', format='wav')
        stream.playrec(inp_fh, out=out_fh, blocking=True)

    out_fh.seek(0); inp_fh.seek(0)
    assert_soundfiles_equal(inp_fh, out_fh, preamble, dtype)

def test_stdin_stdout_loopback(random_soundfile_input, devargs):
    try:
        from shlex import quote as shlex_quote
    except ImportError:
        from pipes import quote as shlex_quote

    inp_fh, preamble, dtype = random_soundfile_input

    inp_fh.name.seek(0)

    proc = subprocess.Popen(('pastream', '-', '-', '-t', 'au', '-f', dtype, '-D', 
        shlex_quote(devargs['device'])), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout = proc.communicate(inp_fh.name.read())[0]
    assert proc.returncode == 0

    inp_fh.name.seek(0)
    with io.BytesIO(stdout) as stdoutf, sf.SoundFile(stdoutf) as out_fh:
        assert_soundfiles_equal(inp_fh, out_fh, preamble, dtype)

def test_offset(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    with ps.DuplexStream(**devargs) as stream:
        for o in (1 << x for x in range(8, 10)):
            out = stream.playrec(inp_fh, 1024, offset=o, blocking=True)
            assert len(out) == 1024 - o
            inp_fh.seek(0)

def test_frames(devargs, random_soundfile_input):
    inp_fh, preamble, dtype = random_soundfile_input

    with ps.DuplexStream(**devargs) as stream:
        i = inp_fh.read(dtype=stream.dtype[1])
        for f in (1 << x for x in range(8, 16)):
            out = stream.playrec(i, f, blocking=True)
            assert len(out) == f
            inp_fh.seek(0)

def test_pad(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    with ps.DuplexStream(**devargs) as stream:
        i = inp_fh.read(dtype=stream.dtype[1])
        for p in (1 << x for x in range(8, 16)):
            out = stream.playrec(i, 1024, pad=p, blocking=True)
            assert len(out) == 1024 + p
            inp_fh.seek(0)

# This tests the condition where frames = -1 and pad >= 0. It makes sure that
# once the txbuff is found to be empty, it goes to 'autoframes' mode and never
# tries to read the txbuff again
def test_autoframes(devargs):

    with ps.DuplexStream(**devargs) as stream:
        txbuffer = ps.RingBuffer(stream.channels[1] * stream.samplesize[1], 1 << 10)
        stream.set_source(txbuffer)
        rxbuffer = ps.RingBuffer(stream.channels[0] * stream.samplesize[0], 1 << 16)
        stream.set_sink(rxbuffer)

        stream._pad = int(stream.samplerate)
        stream._frames = -1
        txbuffer.advance_write_index(1024)
        stream.start()

        # wait to enter autoframes mode
        while not stream._cstream._autoframes:
            time.sleep(0.005)
        assert stream.active

        # write some additional data; this data should be unread at the end of
        # the stream
        txbuffer.advance_write_index(1024)
        stream.wait()
        assert rxbuffer.read_available == 1024 + int(stream.samplerate)
        assert txbuffer.read_available == 1024

def test_frames_pad_offset(devargs):
    with ps.DuplexStream(**devargs) as stream:
        txbuffer = ps.RingBuffer(stream.channels[1] * stream.samplesize[1], 1 << 17)
        stream.set_source(txbuffer)
        rxbuffer = ps.RingBuffer(stream.channels[0] * stream.samplesize[0], 1 << 17)
        stream.set_sink(rxbuffer)

        for f, p, o in [(1 << x + 8, 1 << 16 - x, 1 << x + 6) for x in range(7)]:
            stream._frames = f
            stream._pad = p
            stream._offset = o
            txbuffer.advance_write_index(f)

            stream.start()
            stream.wait()
            assert rxbuffer.read_available == f + p - o

            rxbuffer.flush(); txbuffer.flush()

def test_stream_replay(devargs):
    with ps.OutputStream(**devargs) as stream:
        txbuffer = ps.RingBuffer(stream.channels * stream.samplesize, 8192)
        stream.set_source(txbuffer)

        # Start and let stream finish
        txbuffer.advance_write_index(len(txbuffer))
        stream.start()
        assert stream.wait(2), "Timed out!"
        assert stream.finished

        # Start stream, wait, then abort it
        txbuffer.advance_write_index(len(txbuffer))
        stream.start()
        assert stream.active
        stream.wait(0.001)
        stream.abort()
        assert stream.aborted

        # Start stream then stop it
        txbuffer.advance_write_index(len(txbuffer))
        stream.start()
        assert stream.active
        stream.wait(0.001)
        stream.stop()
        assert stream.stopped

# For testing purposes
class MyException(Exception):
    pass

def test_deferred_exception_handling(devargs):
    stream = ps.OutputStream(**devargs)
    txbuffer = ps.RingBuffer(stream.channels * stream.samplesize, 8192)
    txbuffer.advance_write_index(len(txbuffer))
    stream.set_source(txbuffer)
    with pytest.raises(MyException) as excinfo:
        with stream:
            stream.start()
            stream._set_exception(MyException("BOO-urns!"))
            stream.wait()

def test_threaded_write_deferred_exception_handling(devargs):
    txmsg = "BOO-urns!"
    def writer(stream, ringbuff, loop):
        raise MyException(txmsg)

    stream = ps.OutputStream(**devargs)
    ringbuff = stream.set_source(writer, 8192)
    ringbuff.advance_write_index(len(ringbuff))

    with pytest.raises(MyException) as excinfo:
        with stream:
            stream.start()
            assert stream.wait(0.5), "Timed out!"
    assert str(excinfo.value) == txmsg

def test_threaded_read_deferred_exception_handling(devargs):
    rxmsg = "BOO!"
    def reader(stream, ringbuff):
        raise MyException(rxmsg)

    # A reader exception should also stop the stream
    stream = ps.InputStream(**devargs)
    ringbuff = stream.set_sink(reader, 8192)
    ringbuff.advance_write_index(len(ringbuff))

    with pytest.raises(MyException) as excinfo:
        with stream:
            stream.start()
            assert stream.wait(0.5), "Timed out!"
    assert str(excinfo.value) == rxmsg

def test_frames_raises_underflow(devargs):
    stream = ps.DuplexStream(**devargs)
    txbuffer = ps.RingBuffer(stream.samplesize[1] * stream.channels[1], 8192)
    stream._frames = 9000

    stream.set_source(txbuffer)
    txbuffer.advance_write_index(len(txbuffer))
    with pytest.raises(ps.BufferEmpty) as excinfo:
        with stream:
            stream.start()
            stream.wait()
