import pastream as ps
import soundfile as sf


def set_source(stream, buffer):
    if not isinstance(buffer, ps.RingBuffer):
        buffer = ps._LinearBuffer(stream._cstream.config.txElementSize, buffer)
        buffer.advance_write_index(len(buffer))
    
    # Set the source buffer
    stream._cstream.txbuffer = ps._ffi.cast('PaUtilRingBuffer*', buffer._ptr)
    # Keep buffer alive in cffi land
    stream._txbuffer = buffer

def set_sink(stream, buffer):
    if not isinstance(buffer, ps.RingBuffer):
        buffer = ps._LinearBuffer(stream._cstream.config.rxElementSize, buffer)
        buffer.advance_write_index(len(buffer))
    stream._cstream.rxbuffer = ps._ffi.cast('PaUtilRingBuffer*', buffer._ptr)
    stream._rxbuffer = buffer

def play(stream, buffer):
    stream.reset()
    set_source(stream, buffer)
    stream.start()
    
def rec(stream, buffer):
    stream.reset()
    set_sink(stream, buffer)
    stream.start()

def playrec(stream, ibuffer, obuffer):
    stream.reset()
    set_sink(stream, ibuffer)
    set_source(stream, obuffer)
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
    stream.reset()
    txbuffer = ps.RingBuffer(stream._cstream.config.txElementSize, buffersize)
    set_source(stream, txbuffer)

    if stream.isduplex:
        dtype = stream.dtype[1]
    else:
        dtype = stream.dtype
    
    with sf.SoundFile(ofile) as filehandle:
        # Pre-fill out buffer
        availframes, readframes = read_into(txbuffer, filehandle.buffer_read_into, buffersize, dtype)
        
        stream.start()
        while stream.active and readframes == availframes:
            availframes, readframes = read_into(txbuffer, filehandle.buffer_read_into, buffersize, stream.dtype)

def recfile(stream, ifile, buffersize=2048):
    stream.reset()
    rxbuffer = ps.RingBuffer(stream._cstream.config.rxElementSize, buffersize)
    set_sink(stream, rxbuffer)

    if stream.isduplex:
        dtype = stream.dtype[0]
        channels = stream.channels[0]
    else:
        dtype = stream.dtype
        channels = stream.channels
    
    with sf.SoundFile(ifile, 'w', stream.samplerate, channels) as ifilehandle:
        stream.start()
        while stream.active:
            read_from(rxbuffer, ifilehandle.buffer_write, buffersize, dtype)

        stream.stop()    
        read_from(rxbuffer, ifilehandle.buffer_write, buffersize, dtype)
        
def playrecfile(stream, ifile, ofile, buffersize=2048):
    stream.reset()
    txbuffer = ps.RingBuffer(stream._cstream.config.txElementSize, buffersize)
    set_source(stream, txbuffer)
    
    rxbuffer = ps.RingBuffer(stream._cstream.config.rxElementSize, buffersize)
    set_sink(stream, rxbuffer)
    
    with sf.SoundFile(ofile) as ofilehandle:
        with sf.SoundFile(ifile, 'w', int(stream.samplerate), stream.channels[0]) as ifilehandle:
            # Pre-fill out buffer
            availframes, readframes = read_into(txbuffer, ofilehandle.buffer_read_into, buffersize, stream.dtype[1])
            
            stream.start()
            while stream.active and readframes == availframes:
                availframes, readframes = read_into(txbuffer, ofilehandle.buffer_read_into, buffersize, stream.dtype[1])
                read_from(rxbuffer, ifilehandle.buffer_write, buffersize, stream.dtype[0])

            stream.stop()    
            read_from(rxbuffer, ifilehandle.buffer_write, buffersize, stream.dtype[0])