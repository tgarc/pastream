#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"


int callback(
    const void* in_data, 
    void* out_data, 
    unsigned long frame_count, 
    const PaStreamCallbackTimeInfo* timeInfo, 
    PaStreamCallbackFlags status,
    void *user_data) 
{
    unsigned long frames_left = frame_count, offset = 0;
    ring_buffer_size_t oframes, iframes;
    Py_PaBufferedStream *stream = (Py_PaBufferedStream *) user_data;

#ifdef PYPA_DEBUG
    PaTime timedelta = timeInfo->currentTime - stream->callbackInfo->lastTime;

    if (stream->callbackInfo->call_count == 0) {
        stream->callbackInfo->min_dt = timeInfo->currentTime;
        stream->callbackInfo->max_dt = 0;
    }
    else if (timedelta > stream->callbackInfo->max_dt)
        stream->callbackInfo->max_dt = timedelta;
    if ( timedelta > 0 ) {
        if (timedelta < stream->callbackInfo->min_dt)
            stream->callbackInfo->min_dt = timedelta;
        if ( stream->callbackInfo->call_count > 0 )
            stream->callbackInfo->period[(stream->callbackInfo->call_count-1) % MEASURE_LEN] = timedelta;
        stream->callbackInfo->call_count++;
    }
#endif // PYPA_DEBUG

    switch (status) {
        case paInputUnderflow :
            stream->inputUnderflows++;
            break;
        case paInputOverflow :
            stream->inputOverflows++;
            break;
        case paOutputUnderflow :
            stream->outputUnderflows++;
            break;
        case paOutputOverflow :
            stream->outputOverflows++;
            break;
    }

    stream->status |= status;
    if ( status & 0xF ) {
        stream->xruns++;
        if (status & stream->abort_on_xrun) {
          strcpy(stream->errorMsg, "XRunError");
          return stream->last_callback = paAbort;
        }
    }

    // exit point (1) We've surpassed nframes: this is our last callback
    if ( stream->nframes && stream->frame_count + frames_left >= stream->nframes ) {
        frames_left = stream->nframes - stream->frame_count;
        stream->last_callback = paComplete;
    }

    if (stream->duplexity & O_MODE) {
        oframes = PaUtil_ReadRingBuffer(stream->txbuff, out_data, frames_left);

        // We're done reading frames! Or the writer was too slow; either way,
        // finish up by adding some zero padding.
        if (oframes < frames_left) {
            // Fill the remainder of the output buffer with zeros
            memset((unsigned char *) out_data + oframes*stream->txbuff->elementSizeBytes,
                   0, 
                   (frame_count - oframes)*stream->txbuff->elementSizeBytes);

            if ( !stream->nframes ) {
                // Figure out how much additional padding to insert and set nframes
                // equal to it
                stream->_nframesIsUnset = 1;
                stream->nframes = stream->frame_count + oframes + stream->padding;
                // exit point (2) We don't want to do an unncessary callback; end here
                if ( stream->frame_count + frames_left >= stream->nframes ) {
                    frames_left = stream->nframes - stream->frame_count;
                    stream->last_callback = paComplete;
                }
            }
            else if ( !stream->_nframesIsUnset ) {
                strcpy(stream->errorMsg, "TransmitBufferEmpty");
                stream->frame_count += oframes;
                return stream->last_callback = paAbort;
            }
        }
    }

    if ( stream->duplexity & I_MODE && stream->frame_count + frames_left > stream->offset ) {
        if ( stream->frame_count < stream->offset ) {
            offset = stream->offset - stream->frame_count;
            frames_left -= offset;
            in_data = (unsigned char *) in_data + offset*stream->rxbuff->elementSizeBytes;
        }

        iframes = PaUtil_WriteRingBuffer(stream->rxbuff, (void *) in_data, frames_left);
        if ( iframes < frames_left ) {
            strcpy(stream->errorMsg, "ReceiveBufferFull");
            stream->frame_count += iframes;
            return stream->last_callback = paAbort;
        }
    }

    stream->callbackInfo->lastTime = timeInfo->currentTime;
    stream->frame_count += frame_count;
    return stream->last_callback;
}
