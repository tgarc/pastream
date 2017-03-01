typedef enum duplexity {
  I_MODE = 1,
  O_MODE,
  IO_MODE
} duplexity_t;

typedef struct Py_PsBufferedStream {
  PaStreamCallbackFlags status;
  unsigned long frame_count;
  unsigned long nframes;
  unsigned long padframes;
  duplexity_t duplexity;
  PaUtilRingBuffer* rxq;
  PaUtilRingBuffer* txq;
  char errorMsg[120];
  PaTime max_dt, min_dt, lastTime;
} Py_PsBufferedStream;

int callback(const void* in_data, void* out_data, 
             unsigned long frame_count, 
             const PaStreamCallbackTimeInfo* timeInfo, 
             PaStreamCallbackFlags status, 
             void *user_data);
