#define MAX_MESSAGE_LEN 128


typedef struct Py_PaBufferedStream {
    PaStreamCallbackFlags status;
    PaStreamCallbackFlags abort_on_xrun;
    PaStreamCallbackTimeInfo* lastTime;
    int last_callback;
    int __nframesIsUnset;         // Internal only
    long nframes;                 // Number of frames to play/record
                                  // (0 means play until empty, -1 means play/rec indefinitely)
    long pad;                     // Number of zero frames to pad the playback with
                                  // (< 0 means to pad playback to match nframes)
    unsigned long xruns;
    unsigned long inputOverflows, inputUnderflows;
    unsigned long outputOverflows, outputUnderflows;
    unsigned long frame_count;    // Number of frames successfully processed
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    PaUtilRingBuffer* rxbuff;     // Receive buffer
    PaUtilRingBuffer* txbuff;     // Transmit buffer
    char errorMsg[MAX_MESSAGE_LEN];  // Reserved for errors raised in the audio callback
} Py_PaBufferedStream;

// call once for initialization
void init_stream(
    Py_PaBufferedStream *stream, 
    PaStreamCallbackFlags abort_on_xrun, 
    long nframes,
    long pad,
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
