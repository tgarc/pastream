#define MAX_MESSAGE_LEN 128
/* #define PYPA_DEBUG      1 */


typedef struct Py_PaStream {
    PaStreamCallbackFlags status;
    PaStreamCallbackTimeInfo lastTime;
    int last_callback;
    unsigned char loop;
    unsigned long long frame_count;        // Number of frames successfully processed
    long long frames;             // Number of frames to play/record (-1 means infinite)
    long pad;                     // Number of zero frames to pad the playback with
                                  // (< 0 means to pad playback whenever buffer is empty)
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    unsigned long xruns;
    ring_buffer_size_t txElementSize;
    unsigned long inputOverflows, inputUnderflows;
    unsigned long outputOverflows, outputUnderflows;
    PaUtilRingBuffer* rxbuffer;     // Receive buffer
    PaUtilRingBuffer* txbuffer;     // Transmit buffer
    char errorMsg[MAX_MESSAGE_LEN];  // Reserved for errors raised in the audio callback
    unsigned char _autoframes;   // Internal use only
} Py_PaStream;

// :sigh: Visual Studio doesn't support designated initializors
const Py_PaStream Py_PaStream_default = {
    0,
    { 0 },
    0,
    paContinue,
    0,
    -1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    (PaUtilRingBuffer*) NULL,
    (PaUtilRingBuffer*) NULL,
    { '\0' },
    0
};

// Call once for initialization
void init_stream(Py_PaStream *stream);

// Call before re-starting the stream
void reset_stream(Py_PaStream *stream);

int callback
(
    const void* in_data,
    void* out_data,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags status,
    void *user_data
);
