#define MAX_MESSAGE_LEN 128


typedef struct Py_PaStream {
    PaStreamCallbackFlags status;
    PaStreamCallbackTimeInfo lastTime;
    int last_callback;
    unsigned char __autoframes;   // Internal use only
    long long frames;             // Number of frames to play/record (-1 means infinite)
    long pad;                     // Number of zero frames to pad the playback with
                                  // (< 0 means to pad playback whenever buffer is empty)
    unsigned long xruns;
    unsigned long inputOverflows, inputUnderflows;
    unsigned long outputOverflows, outputUnderflows;
    unsigned long long frame_count;        // Number of frames successfully processed
    unsigned long offset;         // Number of frames to skip from beginning of recordings
    PaUtilRingBuffer* rxbuff;     // Receive buffer
    PaUtilRingBuffer* txbuff;     // Transmit buffer
    char errorMsg[MAX_MESSAGE_LEN];  // Reserved for errors raised in the audio callback
} Py_PaStream;

const Py_PaStream Py_PaStream_default = {
    .last_callback = paContinue,
    .status = 0,
    .frame_count = 0,
    .frames = -1,
    .pad = 0,
    .offset = 0,
    .rxbuff = NULL,
    .txbuff = NULL,
    .xruns = 0,
    .inputUnderflows = 0,
    .inputOverflows = 0,
    .outputUnderflows = 0,
    .outputOverflows = 0,
    .__autoframes = 0
};


// call once for initialization
void init_stream(Py_PaStream *stream);

// Call before re-starting the stream
void reset_stream(Py_PaStream *stream);

int callback(const void* in_data, void* out_data,
             unsigned long frame_count,
             const PaStreamCallbackTimeInfo* timeInfo,
             PaStreamCallbackFlags status,
             void *user_data);
