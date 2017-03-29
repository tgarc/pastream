#define MAX_MESSAGE_LEN 128


typedef enum duplexity_t {
    I_MODE = 1,
    O_MODE,
    IO_MODE
} duplexity_t;

typedef struct Py_PaBufferedStream {
    PaStreamCallbackFlags status;
    PaStreamCallbackFlags abort_on_xrun;
    PaTime lastTime;
    int last_callback;
    int _nframesIsUnset;          // Internal only: must be initialized to 0
    unsigned long xruns;
    unsigned long inputOverflows, inputUnderflows;
    unsigned long outputOverflows, outputUnderflows;
    unsigned long frame_count;    // Number of frames successfully processed
    unsigned long call_count;     // Number of times callback was called
    unsigned long nframes;        // Number of frames to play/record (0 means unlimited)
    unsigned long padding;        // Number of zero frames to pad the input with
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    duplexity_t duplexity;        
    PaUtilRingBuffer* rxbuff;     // Receive buffer
    PaUtilRingBuffer* txbuff;     // Transmit buffer
    char errorMsg[MAX_MESSAGE_LEN];           // Reserved for errors raised in the audio callback
} Py_PaBufferedStream;

int callback(const void* in_data, void* out_data, 
             unsigned long frame_count, 
             const PaStreamCallbackTimeInfo* timeInfo, 
             PaStreamCallbackFlags status, 
             void *user_data);
