#define MAX_MESSAGE_LEN 128


typedef struct Py_PaBufferedStream {
    PaStreamCallbackFlags status;
    unsigned char allow_drops;    // Allow dropping of rxbuff frames
    PaStreamCallbackTimeInfo* lastTime;
    int last_callback;
    unsigned char __autoframes;   // Internal use only
    long frames;                  // Number of frames to play/record (0 means infinite)
    long pad;                     // Number of zero frames to pad the playback with
                                  // (< 0 means to pad playback whenever buffer is empty)
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
    unsigned char allow_drops,
    long frames,
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
