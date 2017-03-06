"""
Loopback tests for pastream.
"""
from __future__ import print_function
import os, platform
import pytest
import time
import tempfile
import numpy as np
import numpy.testing as npt
import soundfile as sf
import sounddevice as sd
import pastream as ps
import pa_ringbuffer


# Set up the platform specific device
system = platform.system()
if system == 'Windows':
    DEVICE_KWARGS = { 'device': 'ASIO4ALL v2, ASIO', 'dtype': 'int24',
                      'blocksize': 512, 'channels': 8, 'samplerate':48000 }
elif system == 'Darwin':
    raise Exception("Currently no support for Mac devices")
else:
    # This is assuming you're using the ALSA device set up by etc/.asoundrc
    DEVICE_KWARGS = { 'device': 'aloop_duplex', 'dtype': 'int32', 'blocksize': 512,
                      'channels': 1, 'samplerate': 48000, 'delay': 2048 }

#DEVICE_KWARGS = { 'device': "miniDSP ASIO Driver, ASIO", 'dtype': 'int24', 'blocksize': 512, 'channels': 8, 'samplerate': 48000 }

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
    inpblocks = xf.blocks(blocksize, dtype=dtype, always_2d=True)
    unsigned_dtype = dtype if dtype.startswith('u') else 'u'+dtype
    for i,inpblk in enumerate(inpblocks):
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
    outblocks = out_fh.blocks(blocksize, dtype=dtype, always_2d=True)
    unsigned_dtype = 'u' + dtype.lstrip('u')
    inpblk = np.zeros((blocksize, inp_fh.channels), dtype=dtype)
    for outblk in outblocks:
        readframes = inp_fh.buffer_read_into(inpblk[:len(outblk)], dtype=dtype)

        inp = inpblk[:readframes].view(unsigned_dtype)
        out = outblk.view(unsigned_dtype)

        npt.assert_array_equal(inp, out, "Loopback data mismatch")
        mframes += readframes

    print("Matched %d of %d frames; Initial delay of %d frames; %d frames truncated" 
          % (mframes, len(inp_fh), delay, len(inp_fh) - inp_fh.tell()))

def assert_blockstream_equal(inp_fh, preamble, **kwargs):
    devargs = dict(DEVICE_KWARGS)
    if 'delay' in devargs: del devargs['delay']
    devargs.update(kwargs)

    stream = ps.SoundFileStream(inp_fh, **devargs)

    # 'tee' the transmit queue writer so that we can recall any input and match
    # it to the output. We make it larger than usual (4M frames) to allow for
    # extra slack
    inpbuff = pa_ringbuffer.RingBuffer(stream.txbuff.elementsize, len(stream.txbuff) * 2)
    writer = stream.txbuff.write
    def teewrite(buff, size=-1):
        nframes1 = writer(buff, size)
        nframes2 = inpbuff.write(buff, nframes1)
        assert nframes1 == nframes2, "Ran out of temporary buffer space. Use a larger buffersize"
        return nframes1
    stream.txbuff.write = teewrite

    delay = -1
    found_delay = False
    unsigned_dtype = 'u%d'%stream.samplesize[1]
    nframes = mframes = 0
    inframes = np.zeros((stream.blocksize, stream.channels[1]), dtype=stream.dtype[1])
    for outframes in stream.blockstream(always_2d=True):
        if not found_delay:
            matches = outframes[:, 0].view(unsigned_dtype) == preamble
            if np.any(matches): 
                found_delay = True
                nonzeros = np.where(matches)[0]
                outframes = outframes[nonzeros[0]:]
                nframes += nonzeros[0]
                delay = nframes
        if found_delay:
            readframes = inpbuff.read(inframes, len(outframes))
            inp = inframes[:readframes].view(unsigned_dtype)
            out = outframes[:readframes].view(unsigned_dtype)

            npt.assert_array_equal(inp, out, "Loopback data mismatch")
            mframes += readframes
        nframes += len(outframes)
    assert delay != -1, "Preamble not found or was corrupted"

    stats = mframes, nframes, delay, inpbuff.read_available, stream._rmisses
    print("Matched %d of %d frames; Initial delay of %d frames; %d frames truncated; %d misses" 
          % stats)
    return stats

def gen_random(rdm_fh, nseconds, elementsize):
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

    preamble = np.zeros((rdm_fh.samplerate//10, rdm_fh.channels), dtype=np.int32)
    preamble[:] = (PREAMBLE >> shift) << shift
    rdm_fh.write(preamble)

    for i in range(nseconds):
        pattern = np.random.randint(minval, maxval+1, (rdm_fh.samplerate, rdm_fh.channels)) << shift
        rdm_fh.write(pattern.astype(np.int32))

@pytest.fixture
def random_soundfile_input(scope='module'):
    elementsize = _dtype2elementsize[DEVICE_KWARGS['dtype']]

    rdmf = tempfile.TemporaryFile()
    rdm_fh = sf.SoundFile(rdmf, 'w+', 
                          DEVICE_KWARGS['samplerate'],
                          DEVICE_KWARGS['channels'], 
                          'PCM_'+['8','16','24','32'][elementsize-1],
                          format='wav')

    with rdm_fh:
        gen_random(rdm_fh, 1, elementsize)
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

def test_blockstream_loopback(random_soundfile_input):
    inp_fh, preamble, dtype = random_soundfile_input
    assert_blockstream_equal(inp_fh, preamble, dtype=dtype)

def test_soundfile_loopback(random_soundfile_input, devargs):
    if 'delay' in devargs: del devargs['delay']
    inp_fh, preamble, dtype = random_soundfile_input

    devargs['dtype'] = dtype
    devargs['sfkwargs'] = {'format': 'wav'}

    outf = tempfile.TemporaryFile()
    with ps.SoundFileStream(inp_fh, outf, **devargs) as stream:
        stream.wait()

    outf.seek(0); inp_fh.seek(0)
    with sf.SoundFile(outf) as out_fh:
        assert_soundfiles_equal(inp_fh, out_fh, preamble, dtype)

def test_padding_offset_nframes(random_soundfile_input, devargs):
    if 'delay' not in devargs: return
    inp_fh, preamble, dtype = random_soundfile_input

    pad = offset = devargs.pop('delay')

    # If we offset and pad the recording using a known fixed delay we should
    # have an *exact* match
    mframes, nframes, delay = assert_blockstream_equal(inp_fh, preamble, dtype=dtype, pad=pad, offset=offset, nframes=8192)[:3]
    assert nframes == mframes and delay == 0 and nframes == 8192

    # Using offset only should drop 'offset' frames from the recording
    inp_fh.seek(0)
    mframes, nframes = assert_blockstream_equal(inp_fh, preamble, dtype=dtype, offset=offset)[:2]
    assert nframes == (len(inp_fh) - offset)

# For testing purposes
class MyException(Exception):
    pass

def test_deferred_exception_handling(devargs):
    if 'delay' in devargs: del devargs['delay']

    stream = ps.BufferedStream(buffersize=8192, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
    with pytest.raises(MyException) as excinfo:
        with stream:
            stream._set_exception(MyException("BOO-urns!"))
            stream.wait()

def test_threaded_write_deferred_exception_handling(devargs):
    if 'delay' in devargs: del devargs['delay']

    txmsg = "BOO-urns!"
    def qwriter(stream, ringbuff):
        raise MyException(txmsg)

    stream = ps.ThreadedStream(buffersize=8192, qwriter=qwriter, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
    with pytest.raises(MyException) as excinfo:
        with stream: stream.wait()
    assert str(excinfo.value) == txmsg

def test_threaded_read_deferred_exception_handling(devargs):
    if 'delay' in devargs: del devargs['delay']

    rxmsg = "BOO!"
    def qreader(stream, ringbuff):
        raise MyException(rxmsg)

    # A reader exception should also stop the stream
    stream = ps.ThreadedStream(buffersize=8192, qreader=qreader, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )
    with pytest.raises(MyException) as excinfo:
        with stream: stream.wait()
    assert str(excinfo.value) == rxmsg

def test_valid_stream_checker(devargs):
    if 'delay' in devargs: del devargs['delay']

    stream = ps.BufferedStream(buffersize=8192, **devargs)
    stream.txbuff.write( bytearray(len(stream.txbuff)*stream.txbuff.elementsize) )

    assert stream._valid == True
    stream.start()
    assert stream._valid == True
    stream.stop()
    assert stream._valid == True
    stream.close()
    assert stream._valid == False
