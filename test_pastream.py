"""
Loopback tests for pastream.
"""
from __future__ import print_function
import os, sys
import numpy as np
import soundfile as sf
import pytest
import numpy.testing as npt
import time
import tempfile
import platform
import pastream as ps
import pa_ringbuffer


# Set up the platform specific device
system = platform.system()
if system == 'Windows':
    DEVICE_KWARGS = {'device': 'ASIO4ALL v2, ASIO', 'dtype': 'int24', 'blocksize': 512, 'channels': 8, 'samplerate':48000}
elif system == 'Darwin':
    raise Exception("Currently no support for Mac devices")
else:
    # This is assuming you're using the ALSA device set up by etc/.asoundrc
    DEVICE_KWARGS = {'device': 'aduplex', 'dtype': 'int32', 'blocksize': 512, 'channels': 8, 'samplerate':48000}

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

def assert_loopback_equal(inp_fh, preamble, **kwargs):
    devargs = dict(DEVICE_KWARGS)
    devargs.update(kwargs)

    stream = ps.SoundFileStream(inp_fh, **devargs)

    # 'tee' the transmit queue writer so that we can recall any input and match
    # it to the output. We make it larger than usual (4M frames) to allow for
    # extra slack
    inpbuff = pa_ringbuffer.RingBuffer(stream.framesize[1], 1 << 24)
    stream.txq.__write = stream.txq.write
    def teewrite(*args, **kwargs):
        nframes1 = inpbuff.write(*args, **kwargs)
        nframes2 = stream.txq.__write(*args, **kwargs)
        assert nframes1 == nframes2, "Ran out of temporary buffer space. Use a larger qsize"
        return nframes1
    stream.txq.write = teewrite

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

    print("Matched %d of %d frames; Initial delay of %d frames" % (mframes, nframes, delay))

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

def test_loopback():
    elementsize = _dtype2elementsize[DEVICE_KWARGS['dtype']]

    rdmf = tempfile.TemporaryFile()
    rdm_fh = sf.SoundFile(rdmf, 'w+', DEVICE_KWARGS['samplerate'], DEVICE_KWARGS['channels'], 'PCM_'+['8', '16','24','32'][elementsize-1], format='wav')

    gen_random(rdm_fh, 5, elementsize)
    rdm_fh.seek(0)

    dtype = DEVICE_KWARGS['dtype']
    if DEVICE_KWARGS['dtype'] == 'int24': 
        # Tell the OS it's a 32-bit stream and ignore the extra zeros
        # because 24 bit streams are annoying to deal with
        dtype = 'int32' 

    shift = 8*(4-elementsize)
    assert_loopback_equal(rdm_fh, (PREAMBLE>>shift)<<shift, dtype=dtype)
