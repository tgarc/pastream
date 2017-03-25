typedef enum duplexity_t {
    I_MODE = 1,
    O_MODE,
    IO_MODE
} duplexity_t;

typedef struct Py_PaCallbackInfo {
    PaTime max_dt, min_dt, lastTime;
    long min_frame_count;
    unsigned long call_count;
    unsigned long inputOverflows, inputUnderflows;
    unsigned long outputOverflows, outputUnderflows;
} Py_PaCallbackInfo;

typedef struct Py_PaBufferedStream {
    PaStreamCallbackFlags status;
    int last_callback;
    int abort_on_xrun;
    int _nframesIsUnset;
    unsigned long xruns;
    unsigned long frame_count;
    unsigned long nframes;        // Number of frames to play/record (0 means unlimited)
    unsigned long padding;        // Number of zero frames to pad the input with
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    duplexity_t duplexity;        
    PaUtilRingBuffer* rxbuff;     // Receive buffer
    PaUtilRingBuffer* txbuff;     // Transmit buffer
    char errorMsg[120];           // Reserved for errors raised in the audio callback
    Py_PaCallbackInfo* callbackInfo;
} Py_PaBufferedStream;

int callback(const void* in_data, void* out_data, 
             unsigned long frame_count, 
             const PaStreamCallbackTimeInfo* timeInfo, 
             PaStreamCallbackFlags status, 
             void *user_data);
