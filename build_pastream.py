from cffi import FFI
import pa_ringbuffer

ffibuilder = FFI()
ffibuilder.cdef(r"""
/* From portaudio.h: */

typedef double PaTime;
typedef struct PaStreamCallbackTimeInfo{
    PaTime inputBufferAdcTime;
    PaTime currentTime;
    PaTime outputBufferDacTime;
} PaStreamCallbackTimeInfo;
typedef unsigned long PaStreamCallbackFlags;
#define paInputUnderflow  0x00000001
#define paInputOverflow   0x00000002
#define paOutputUnderflow 0x00000004
#define paOutputOverflow  0x00000008
#define paPrimingOutput   0x00000010

""")

ffibuilder.cdef(pa_ringbuffer.cdef())

ffibuilder.cdef(open('src/py_pastream.h').read())

ffibuilder.set_source(
    '_py_pastream',
    open('src/py_pastream.c').read(),
    include_dirs=['src', 'portaudio/include', 'portaudio/src/common'],
    sources=['portaudio/src/common/pa_ringbuffer.c'],
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
