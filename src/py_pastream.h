#define MAX_MESSAGE_LEN 128


typedef struct Py_PaBufferedStream {
    PaStreamCallbackFlags status;
    PaStreamCallbackFlags abort_on_xrun;
    int keep_alive;
    /* int paused; */
    PaStreamCallbackTimeInfo* lastTime;
    int last_callback;
    int __nframesIsUnset;         // Internal only
    unsigned long xruns;
    unsigned long inputOverflows, inputUnderflows;
    unsigned long outputOverflows, outputUnderflows;
    unsigned long frame_count;    // Number of frames successfully processed
    unsigned long call_count;     // Number of times callback was called
    unsigned long nframes;        // Number of frames to play/record (0 means unlimited)
    unsigned long padding;        // Number of zero frames to pad the input with
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    PaUtilRingBuffer* rxbuff;     // Receive buffer
    PaUtilRingBuffer* txbuff;     // Transmit buffer
    char errorMsg[MAX_MESSAGE_LEN];           // Reserved for errors raised in the audio callback
} Py_PaBufferedStream;

// call once for initialization
void init_stream(
    Py_PaBufferedStream *stream, 
    int keep_alive, 
    PaStreamCallbackFlags abort_on_xrun, 
    unsigned long nframes,
    unsigned long padding,
    unsigned long offset,
    PaUtilRingBuffer *rxbuff,
    PaUtilRingBuffer *txbuff);

// Call before re-starting the stream
void reset_stream(Py_PaBufferedStream *stream);

int callback(const void* in_data, void* out_data, 
             unsigned long frame_count, 
             const PaStreamCallbackTimeInfo* timeInfo, 
             PaStreamCallbackFlags status, 
             void *user_data);
