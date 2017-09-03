"""
Loopback tests for pastream.
"""
from __future__ import print_function
import os, platform, subprocess, io
import pytest
import time
import tempfile
import numpy as np
import numpy.testing as npt
import soundfile as sf
import sounddevice as sd
import pastream as ps


# Set up the platform specific device
system = platform.system()
if system == 'Windows':
    DEVICE_KWARGS = { 'device': 'ASIO4ALL v2, ASIO', 'dtype': 'int24',
                      'channels': 8, 'samplerate':48000 }
elif system == 'Darwin':
    raise Exception("Currently no test support for OSX")
else:
    # This is assuming you're using the ALSA device set up by .asoundrc config
    # file in the root of the repository
    DEVICE_KWARGS = { 'device': 'aloop_duplex', 'dtype': 'int32', 'channels':
                      2, 'samplerate': 48000 }

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

PREAMBLE = 0x7FFFFFFF # Value used for the preamble sequence (before appropriate shifting for dtype)

_dtype2elementsize = dict(int32=4,int24=3,int16=2,int8=1)
vhex = lambda x: np.vectorize(('{:#0%dx}'%(x.dtype.itemsize*2+2)).format)
tohex = lambda x: vhex(x.view('u%d'%x.dtype.itemsize))

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

def assert_chunks_equal(inp_fh, preamble, compensate_delay=False, chunksize=None, outtype=None, **kwargs):
    devargs = dict(DEVICE_KWARGS)
    devargs.update(kwargs)

    inpf2 = sf.SoundFile(inp_fh.name.name)

    inp_fh.seek(0)
    with ps.SoundFileDuplexStream(outf=inp_fh, **devargs) as stream:
        delay = -1
        found_delay = False
        unsigned_dtype = 'u%d' % stream.samplesize[1]
        frames = mframes = 0
        inframes = np.zeros((len(stream.rxbuff), stream.channels[1]), dtype=stream.dtype[1])

        if outtype is not None:
            out = outtype(len(stream.rxbuff) * stream.rxbuff.elementsize)
        else:
            out = None

        t = looptime = 0
        for i, outframes in enumerate(stream.chunks(chunksize, out=out), start=1):
            if outtype is not None:
                outframes = np.frombuffer(outframes, dtype=stream.dtype[0]).reshape(-1, stream.channels[0])

            dt = 1e3*(time.time() - t)
            if i > 1: looptime += dt
            t = time.time()

            if not found_delay:
                matches = outframes.view(unsigned_dtype) == preamble
                if np.any(matches):
                    found_delay = True
                    mindices = np.where(matches)[0]
                    frames += mindices[0]
                    delay = frames
                    if compensate_delay: stream.pad = delay
                    outframes = outframes[mindices[0]:]

            if found_delay:
                readframes = inpf2.buffer_read_into(inframes[:len(outframes)], dtype=stream.dtype[1])
                inp = inframes[:readframes].view(unsigned_dtype)
                out = outframes[:readframes].view(unsigned_dtype)
                npt.assert_array_equal(inp, out, "Loopback data mismatch")
                mframes += readframes

            frames += len(outframes)

        assert delay != -1, "Preamble not found or was corrupted"

    stats = mframes, frames, delay, len(inpf2) - inpf2.tell(), looptime / i, stream._rmisses
    print("Matched %d of %d frames; Initial delay of %d frames; %d frames truncated; %f interlooptime; %d misses"
          % stats)
    return stats

def gen_random(nseconds, samplerate, channels, elementsize):
    """
    Generates a uniformly random integer signal ranging between the
    minimum and maximum possible values as defined by `elementsize`. The random
    signal is preceded by a constant level equal to the maximum positive
    integer value for 100ms or N=sampling_rate/10 samples (the 'preamble')
    which can be used in testing to find the beginning of a recording.

    nseconds - how many seconds of data to generate
    elementsize - size of each element (single sample of a single frame) in bytes
    """
    shift = 8*(4-elementsize)
    minval = -(0x80000000>>shift)
    maxval = 0x7FFFFFFF>>shift

    preamble = np.zeros((samplerate//10, channels), dtype=np.int32)
    preamble[:] = (PREAMBLE >> shift) << shift
    yield preamble

    for i in range(nseconds):
        # sequential pattern for debugging
        # pattern = np.arange(i*samplerate*channels,
        #                     (i+1)*samplerate*channels)\
        #                     .reshape(samplerate, channels) << shift
        pattern = np.random.randint(minval, maxval+1, (samplerate, channels)) << shift
        yield pattern.astype(np.int32)

@pytest.fixture
def random_soundfile_input(tmpdir, scope='session'):
    elementsize = _dtype2elementsize[DEVICE_KWARGS['dtype']]

    # we don't use an actual TemporaryFile because they don't support multiple
    # file handles on windows
    rdmf= open(tempfile.mktemp(dir=str(tmpdir)), 'w+b')
    rdm_fh = sf.SoundFile(rdmf, 'w+',
                          DEVICE_KWARGS['samplerate'],
                          DEVICE_KWARGS['channels'],
                          'PCM_'+['8','16','24','32'][elementsize-1],
                          format='wav')

    with rdm_fh:
        for blk in gen_random(1, rdm_fh.samplerate, rdm_fh.channels, elementsize):
            rdm_fh.write(blk)
        rdm_fh.seek(0)

        dtype = DEVICE_KWARGS['dtype']
        if DEVICE_KWARGS['dtype'] == 'int24':
            # Tell the OS it's a 32-bit stream and ignore the extra zeros
            # because 24 bit streams are annoying to deal with
            dtype = 'int32'

        shift = 8*(4-elementsize)

        yield rdm_fh, (PREAMBLE>>shift)<<shift, dtype

@pytest.fixture
def devargs():
    return dict(DEVICE_KWARGS)

def test_chunks_ndarray_loopback(random_soundfile_input):
    inp_fh, preamble, dtype = random_soundfile_input
    assert_chunks_equal(inp_fh, preamble, dtype=dtype)

def test_chunks_bytearray_loopback(random_soundfile_input):
    inp_fh, preamble, dtype = random_soundfile_input
    assert_chunks_equal(inp_fh, preamble, dtype=dtype, outtype=bytearray)

def test_soundfilestream_loopback(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    devargs['dtype'] = dtype

    outf = tempfile.TemporaryFile()
    with ps.SoundFileDuplexStream(outf, inp_fh, format='wav', **devargs) as stream:
        stream.start()
        stream.wait()

    outf.seek(0); inp_fh.seek(0)
    with sf.SoundFile(outf) as out_fh:
        assert_soundfiles_equal(inp_fh, out_fh, preamble, dtype)

def test_stdin_stdout_loopback(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    devargs['dtype'] = dtype

    inp_fh.name.seek(0)

    proc = subprocess.Popen(('pastream -twav - -tau - -d=%s -f=int32' % devargs['device']).split(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout = io.BytesIO(proc.communicate(inp_fh.name.read())[0])

    inp_fh.name.seek(0)
    with sf.SoundFile(stdout) as out_fh:
        assert_soundfiles_equal(inp_fh, out_fh, preamble, dtype)

def test_offset(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    # Using offset only should drop 'offset' frames from the recording
    offset = 8 # use a minimal offset so we don't drop the original input frames
    mframes, frames = assert_chunks_equal(inp_fh, preamble, dtype=dtype, offset=offset)[:2]
    assert frames == (len(inp_fh) - offset)

def test_frames(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    with ps.DuplexStream(buffersize=1 << 15, **devargs) as stream:
        for f in [1 << x for x in range(8, 16)]:
            stream.frames = f
            stream.txbuff.write(bytearray(stream.frames * stream.txbuff.elementsize))
            stream.start()
            stream.wait()
            assert stream.rxbuff.read_available == stream.frames

            stream.rxbuff.flush()
            stream.txbuff.flush()

def test_pad(random_soundfile_input, devargs):
    inp_fh, preamble, dtype = random_soundfile_input

    # If we compensate for the delay we should have no frames truncated
    mframes, frames, delay, ntrunc = assert_chunks_equal(inp_fh, preamble, dtype=dtype, compensate_delay=True)[:4]
    assert ntrunc == 0

# This tests the condition where frames = -1 and pad >= 0. It makes sure that
# once the txbuff is found to be empty, it goes to 'autoframes' mode and never
# tries to read the txbuff again
def test_autoframes(devargs):
    with ps.DuplexStream(buffersize=1<<17, **devargs) as stream:
        stream.pad = int(stream.samplerate)
        stream.frames = -1
        stream.txbuff.write(bytearray(1024 * stream.txbuff.elementsize))
        stream.start()
        while stream.txbuff.read_available:
            time.sleep(0.01)
        stream.txbuff.write(bytearray(1024 * stream.txbuff.elementsize))
        stream.wait()
        assert stream.txbuff.read_available == 1024
        assert stream.rxbuff.read_available == 1024 + int(stream.samplerate)

def test_frames_pad_offset(devargs):
    with ps.DuplexStream(buffersize=1<<17, **devargs) as stream:
        for f, p, o in [(1 << x + 8, 1 << 16 - x, 1 << x + 6) for x in range(7)]:
            stream.frames = f
            stream.pad = p
            stream.offset = o
            stream.txbuff.write(bytearray(stream.frames * stream.txbuff.elementsize))
            stream.start()
            stream.wait()
            assert stream.rxbuff.read_available == stream.frames + stream.pad - stream.offset

            stream.rxbuff.flush()
            stream.txbuff.flush()

def test_stream_replay(devargs):
    with ps.DuplexStream(**devargs) as stream:
        data = bytearray(len(stream.txbuff)*stream.txbuff.elementsize)

        # Start and let stream finish
        stream.txbuff.write(data)
        stream.start()
        assert stream.wait(2), "Timed out!"
        assert stream.finished
        stream.rxbuff.flush()

        # Start stream, wait, then abort it
        stream.txbuff.write(data)
        stream.start()
        assert stream.active
        stream.rxbuff.flush()
        stream.wait(0.001)
        stream.abort()
        assert stream.aborted
        stream.rxbuff.flush()

        # Start stream then stop it
        stream.txbuff.write(data)
        stream.start()
        assert stream.active
        stream.wait(0.001)
        stream.stop()
        assert stream.stopped
        stream.rxbuff.flush()

# For testing purposes
class MyException(Exception):
    pass

def test_deferred_exception_handling(devargs):
    stream = ps.DuplexStream(buffersize=8192, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
    with pytest.raises(MyException) as excinfo:
        with stream:
            stream.start()
            stream._set_exception(MyException("BOO-urns!"))
            stream.wait()

def test_threaded_write_deferred_exception_handling(devargs):
    txmsg = "BOO-urns!"
    def writer(stream, ringbuff):
        raise MyException(txmsg)

    stream = ps.DuplexStream(buffersize=8192, writer=writer, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
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
    stream = ps.DuplexStream(buffersize=8192, reader=reader, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
    with pytest.raises(MyException) as excinfo:
        with stream:
            stream.start()
            assert stream.wait(0.5), "Timed out!"
    assert str(excinfo.value) == rxmsg

def test_frames_raises_underflow(devargs):
    stream = ps.DuplexStream(buffersize=8192, frames=9000, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
    with pytest.raises(ps.BufferEmpty) as excinfo:
        with stream:
            stream.start()
            stream.wait()
