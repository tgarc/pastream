#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "pastream.h"


int callback(
    const void* in_data, 
    void* out_data, 
    unsigned long frame_count, 
    const PaStreamCallbackTimeInfo* timeInfo, 
    PaStreamCallbackFlags status,
    void *user_data) 
{
  ring_buffer_size_t iframes, oframes;
  Py_PsBufferedStream *stream = (Py_PsBufferedStream *) user_data;
  PaTime timedelta = timeInfo->currentTime - stream->lastTime;

  if (timedelta == timeInfo->currentTime) {
    stream->min_dt = timeInfo->currentTime;
    stream->max_dt = 0;
  }
  else if (timedelta > stream->max_dt) {
    stream->max_dt = timedelta;
  }
  else if (timedelta > 0 && timedelta < stream->min_dt) {
    stream->min_dt = timedelta;
  }

  stream->status |= status;
  if (stream->status&0xF) {
    if (stream->status & paInputUnderflow)  { strcpy(stream->errorMsg, "Input underflow!"); }
    if (stream->status & paInputOverflow)   { strcpy(stream->errorMsg, "Input overflow!"); }
    if (stream->status & paOutputUnderflow) { strcpy(stream->errorMsg, "Output underflow!"); }
    if (stream->status & paOutputOverflow)  { strcpy(stream->errorMsg, "Output overflow!"); }
    return paAbort;
  }

  if (stream->duplexity & I_MODE) {
    iframes = PaUtil_WriteRingBuffer(stream->rxq, in_data, frame_count);
    if (iframes < frame_count) {
      strcpy(stream->errorMsg, "Receive queue is full.");
      return paAbort;
    }
  }

  if (stream->duplexity & O_MODE) {
    oframes = PaUtil_ReadRingBuffer(stream->txq, out_data, frame_count);
    // This is our last callback!
    if (oframes < frame_count) {
      stream->frame_count += oframes;
      return paComplete;
    }
  }

  stream->lastTime = timeInfo->currentTime;
  stream->frame_count += frame_count;
  return paContinue;
}
