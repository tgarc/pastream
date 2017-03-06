typedef enum duplexity {
    I_MODE = 1,
    O_MODE,
    IO_MODE
} duplexity_t;

typedef struct Py_PsCallbackInfo {
    PaTime max_dt, min_dt, lastTime;
    unsigned long call_count;
    unsigned long min_frame_count;
    unsigned long xruns;
} Py_PsCallbackInfo;

typedef struct Py_PsBufferedStream {
    PaStreamCallbackFlags status;
    int completed;                // used to check whether a stream was aborted
    int abort_on_xrun;
    unsigned long frame_count;
    unsigned long nframes;        // Number of frames to play/record (0 means unlimited)
    unsigned long padframes;      // Number of zero frames to pad the input with
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    duplexity_t duplexity;        
    PaUtilRingBuffer* rxq;        // Receive buffer
    PaUtilRingBuffer* txq;        // Transmit buffer
    char errorMsg[120];           // Reserved for errors raised in the audio callback
    Py_PsCallbackInfo* callbackInfo;
} Py_PsBufferedStream;

int callback(const void* in_data, void* out_data, 
             unsigned long frame_count, 
             const PaStreamCallbackTimeInfo* timeInfo, 
             PaStreamCallbackFlags status, 
             void *user_data);
