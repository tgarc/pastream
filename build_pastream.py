from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(r"""
/* From portaudio.h: */

typedef double PaTime;
typedef struct PaStreamCallbackTimeInfo{
    PaTime inputBufferAdcTime;
    PaTime currentTime;
    PaTime outputBufferDacTime;
} PaStreamCallbackTimeInfo;
typedef long ring_buffer_size_t;
typedef unsigned long PaStreamCallbackFlags;
#define paInputUnderflow  0x00000001
#define paInputOverflow   0x00000002
#define paOutputUnderflow 0x00000004
#define paOutputOverflow  0x00000008
#define paPrimingOutput   0x00000010

/* From pa_ringbuffer.h: */

typedef struct PaUtilRingBuffer
{
    ring_buffer_size_t  bufferSize;
    volatile ring_buffer_size_t writeIndex;
    volatile ring_buffer_size_t readIndex;
    ring_buffer_size_t elementSizeBytes;
    char* buffer;
    ...;
} PaUtilRingBuffer;

ring_buffer_size_t PaUtil_InitializeRingBuffer(PaUtilRingBuffer* rbuf, ring_buffer_size_t elementSizeBytes, ring_buffer_size_t elementCount, void* dataPtr);
void PaUtil_FlushRingBuffer(PaUtilRingBuffer* rbuf);
ring_buffer_size_t PaUtil_GetRingBufferWriteAvailable(const PaUtilRingBuffer* rbuf);
ring_buffer_size_t PaUtil_GetRingBufferReadAvailable(const PaUtilRingBuffer* rbuf);
ring_buffer_size_t PaUtil_WriteRingBuffer(PaUtilRingBuffer* rbuf, const void* data, ring_buffer_size_t elementCount);
ring_buffer_size_t PaUtil_ReadRingBuffer(PaUtilRingBuffer* rbuf, void* data, ring_buffer_size_t elementCount);
ring_buffer_size_t PaUtil_GetRingBufferWriteRegions(PaUtilRingBuffer* rbuf, ring_buffer_size_t elementCount, void** dataPtr1, ring_buffer_size_t* sizePtr1, void** dataPtr2, ring_buffer_size_t* sizePtr2);
ring_buffer_size_t PaUtil_AdvanceRingBufferWriteIndex(PaUtilRingBuffer* rbuf, ring_buffer_size_t elementCount);
ring_buffer_size_t PaUtil_GetRingBufferReadRegions(PaUtilRingBuffer* rbuf, ring_buffer_size_t elementCount, void** dataPtr1, ring_buffer_size_t* sizePtr1, void** dataPtr2, ring_buffer_size_t* sizePtr2);
ring_buffer_size_t PaUtil_AdvanceRingBufferReadIndex(PaUtilRingBuffer* rbuf, ring_buffer_size_t elementCount);
""")

ffibuilder.cdef(open('src/py_pastream.h').read())

ffibuilder.set_source(
    '_py_pastream',
    open('src/py_pastream.c').read(),
    include_dirs=['src', 'portaudio/include', 'portaudio/src/common'],
    sources=['portaudio/src/common/pa_ringbuffer.c'],
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
