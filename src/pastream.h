typedef enum duplexity {
  I_MODE = 1,
  O_MODE,
  IO_MODE
} duplexity_t;

typedef struct Py_PsBufferedStream {
  PaStreamCallbackFlags status;
  unsigned long frame_count;
  unsigned long nframes;        // Number of frames to play/record (0 means unlimited)
  unsigned long padframes;      // Number of zero frames to pad the input with
  unsigned long offset;         // Number of frames to skip from beginning of recordings
  duplexity_t duplexity;        
  PaUtilRingBuffer* rxq;        // Receive buffer
  PaUtilRingBuffer* txq;        // Transmit buffer
  char errorMsg[120];           // Reserved for errors raised in the audio callback
  PaTime max_dt, min_dt, lastTime;
} Py_PsBufferedStream;

int callback(const void* in_data, void* out_data, 
             unsigned long frame_count, 
             const PaStreamCallbackTimeInfo* timeInfo, 
             PaStreamCallbackFlags status, 
             void *user_data);
