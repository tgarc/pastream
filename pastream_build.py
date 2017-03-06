from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(r"""
/* From portaudio.h: */

typedef double PaTime;
typedef struct PaStreamCallbackTimeInfo PaStreamCallbackTimeInfo;
typedef unsigned long PaStreamCallbackFlags;
typedef long ring_buffer_size_t;

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
""")

ffibuilder.cdef(open('src/py_pastream.h').read())

ffibuilder.set_source(
    '_pastream',
    open('src/py_pastream.c').read(),
    include_dirs=['src', 'portaudio/include', 'portaudio/src/common'],
    sources=['portaudio/src/common/pa_ringbuffer.c'],
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
