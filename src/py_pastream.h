#define MAX_MESSAGE_LEN 128
// Enable flags with e.g.
// CFLAGS='-DPYPA_DEBUG' pip install .
// #define PYPA_DEBUG
// #define PYPA_WIREMODE

typedef struct Py_PaStreamStats {
    unsigned long frame_count;        // Number of frames successfully processed
    unsigned xruns;
    unsigned inputUnderflows;
    unsigned outputUnderflows;
    unsigned inputOverflows;
    unsigned outputOverflows;
} Py_PaStreamStats;

typedef struct Py_PaStreamConfig {
    unsigned char loop;
    unsigned char keepAlive;
    ring_buffer_size_t rxElementSize;    
    ring_buffer_size_t txElementSize;
} Py_PaStreamConfig;
    
typedef struct Py_PaStream {
    PaUtilRingBuffer *rxbuffer;
    PaUtilRingBuffer *txbuffer;
    PaStreamCallbackTimeInfo lastTime;
    Py_PaStreamStats stats;
    Py_PaStreamConfig config;
} Py_PaStream;

// Call once for initialization
void init_stream(Py_PaStream *stream);

// Call before re-starting the stream
void reset_stream(Py_PaStream *stream);

int callback
(
    void* in_data,
    void* out_data,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags status,
    void *user_data
);
