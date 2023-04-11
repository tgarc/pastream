import pastream as ps
import soundfile as sf


def _set_buffer(stream, buffer, isOutputNotInput):    
    if stream.isduplex:
        elementsize = stream.channels[isOutputNotInput] * stream.samplesize[isOutputNotInput]
    else:
        elementsize = stream.channels * stream.samplesize
    
    # Wrap user buffer
    if not isinstance(buffer, ps.RingBuffer):
        rawbuffer = ps._LinearBuffer(elementsize, buffer)
        rawbuffer.advance_write_index(len(rawbuffer))
    else:
        rawbuffer = buffer
    
    if isOutputNotInput:
        # Set the source buffer
        stream._cstream.txbuffer = ps._ffi.cast('PaUtilRingBuffer*', rawbuffer._ptr)
        # Keep buffer alive in cffi land
        stream._txbuffer = rawbuffer
    else:
        stream._cstream.rxbuffer = ps._ffi.cast('PaUtilRingBuffer*', rawbuffer._ptr)
        stream._rxbuffer = rawbuffer

def set_source(stream, buffer):
    _set_buffer(stream, buffer, 1)
    
def set_sink(stream, buffer):
    _set_buffer(stream, buffer, 0)

def play(stream, buffer):
    _set_buffer(stream, buffer, 1)
    stream.start()
    
def rec(stream, buffer):
    _set_buffer(stream, buffer, 0)
    stream.start()

def playrec(stream, ibuffer, obuffer):
    _set_buffer(stream, ibuffer, 0)
    _set_buffer(stream, obuffer, 1)
    stream.start()
    
def read_into(ringbuffer, buffer_read, nframes, dtype):
    frames, buffregn1, buffregn2 = ringbuffer.get_write_buffers(nframes)
    readframes = buffer_read(buffregn1, dtype=dtype)
    if len(buffregn2):
        readframes += buffer_read(buffregn2, dtype=dtype)
    ringbuffer.advance_write_index(readframes)
    
    return frames, readframes
    
def read_from(ringbuffer, buffer_write, nframes, dtype):
    frames, buffregn1, buffregn2 = ringbuffer.get_read_buffers(nframes)
    buffer_write(buffregn1, dtype=dtype)
    if len(buffregn2):
        buffer_write(buffregn2, dtype=dtype)
    ringbuffer.advance_read_index(frames)
    
    return frames

def playfile(stream, ofile, buffersize=2048):
    elementsize = stream.channels * stream.samplesize
    txbuffer = ps.RingBuffer(elementsize, buffersize)
    _set_buffer(stream, txbuffer, 1)
    
    with sf.SoundFile(ofile) as filehandle:
        # Pre-fill out buffer
        availframes, readframes = read_into(txbuffer, filehandle.buffer_read_into, buffersize, stream.dtype)
        
        stream.start()
        while stream.active and readframes == availframes:
            availframes, readframes = read_into(txbuffer, filehandle.buffer_read_into, buffersize, stream.dtype)

def recfile(stream, ifile, buffersize=2048):
    elementsize = stream.channels[0] * stream.samplesize[0]
    rxbuffer = ps.RingBuffer(elementsize, buffersize)
    _set_buffer(stream, rxbuffer, 0)
    
    with sf.SoundFile(ifile, 'w', stream.samplerate, stream.channels) as ifilehandle:
        stream.start()
        while stream.active:
            read_from(rxbuffer, ifilehandle.buffer_write, buffersize, stream.dtype[0])
                
def playrecfile(stream, ifile, ofile, buffersize=2048):
    elementsize = stream.channels[1] * stream.samplesize[1]
    txbuffer = ps.RingBuffer(elementsize, buffersize)
    _set_buffer(stream, txbuffer, 1)
    
    elementsize = stream.channels[0] * stream.samplesize[0]
    rxbuffer = ps.RingBuffer(elementsize, buffersize)
    _set_buffer(stream, rxbuffer, 0)
    
    with sf.SoundFile(ofile) as ofilehandle:
        with sf.SoundFile(ifile, 'w', int(stream.samplerate), stream.channels[0]) as ifilehandle:
            # Pre-fill out buffer
            availframes, readframes = read_into(txbuffer, ofilehandle.buffer_read_into, buffersize, stream.dtype[1])
            
            stream.start()
            while stream.active and readframes == availframes:
                availframes, readframes = read_into(txbuffer, ofilehandle.buffer_read_into, buffersize, stream.dtype[1])
                read_from(rxbuffer, ifilehandle.buffer_write, buffersize, stream.dtype[0])