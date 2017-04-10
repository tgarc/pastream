#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"


void init_stream(
    Py_PaBufferedStream *stream, 
    int keep_alive, 
    PaStreamCallbackFlags abort_on_xrun, 
    unsigned long nframes,
    unsigned long padding,
    unsigned long offset,
    PaUtilRingBuffer *rxbuff,
    PaUtilRingBuffer *txbuff) 
{
    stream->keep_alive = keep_alive;
    stream->abort_on_xrun = abort_on_xrun;
    stream->nframes = nframes ? (nframes + padding) : 0;
    stream->padding = padding;
    stream->offset = offset;
    stream->rxbuff = rxbuff;
    stream->txbuff = txbuff;
    reset_stream(stream);
};

void reset_stream(Py_PaBufferedStream *stream) {
    memset((void *) stream->lastTime, 0, sizeof(PaStreamCallbackTimeInfo));
    stream->last_callback = paContinue;
    stream->status = 0;
    stream->frame_count = 0;
    stream->call_count = 0;
    stream->errorMsg[0] = '\0';
    stream->xruns = 0;
    stream->inputUnderflows = 0;
    stream->inputOverflows = 0;
    stream->outputUnderflows = 0;
    stream->outputOverflows = 0;
    if ( stream->__nframesIsUnset )
        stream->nframes = 0;
    stream->__nframesIsUnset = 0;
};

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

    ++stream->call_count;
    if ( status & 0xF ) {
        stream->status |= status;
        stream->xruns++;
        if ( status & paInputUnderflow )
            stream->inputUnderflows++;
        if ( status & paInputOverflow )
            stream->inputOverflows++;
        if ( status & paOutputUnderflow )
            stream->outputUnderflows++;
        if ( status & paOutputOverflow )
            stream->outputOverflows++;
        if (status & stream->abort_on_xrun) {
            strcpy(stream->errorMsg, "XRunError");
            return stream->last_callback = paAbort;
        }
    }

    /* if ( stream->paused ) { */
    /*     if (stream->txbuff != NULL) { */
    /*         memset(out_data, 0, frame_count*stream->txbuff->elementSizeBytes); */
    /*     } */
    /*     stream->lastTime = timeInfo->currentTime; */
    /*     return stream->last_callback = paContinue; */
    /* } */

    // exit point (1) We've surpassed nframes: this is our last callback
    if ( stream->nframes && stream->frame_count + frames_left >= stream->nframes ) {
        frames_left = stream->nframes - stream->frame_count;
        stream->last_callback = paComplete;
    }

    if (stream->txbuff != NULL) {
        oframes = PaUtil_ReadRingBuffer(stream->txbuff, out_data, frames_left);

        // We're done reading frames! Or the writer was too slow; either way,
        // finish up by adding some zero padding.
        if (oframes < frames_left) {
            // Fill the remainder of the output buffer with zeros
            memset((unsigned char *) out_data + oframes*stream->txbuff->elementSizeBytes,
                   0, 
                   (frame_count - oframes)*stream->txbuff->elementSizeBytes);

            if ( stream->keep_alive ) {
                ; // pass
            }
            else if ( !stream->nframes ) {
                // Figure out how much additional padding to insert and set nframes
                // equal to it
                stream->__nframesIsUnset = 1;
                stream->nframes = stream->frame_count + oframes + stream->padding;
                // exit point (2) We don't want to do an unncessary callback; end here
                if ( stream->frame_count + frames_left >= stream->nframes ) {
                    if ( stream->frame_count < stream->nframes )
                        frames_left = stream->nframes - stream->frame_count;
                    else
                        frames_left = 0;
                    stream->last_callback = paComplete;
                }
            }
            else if ( !stream->__nframesIsUnset ) {
                strcpy(stream->errorMsg, "TransmitBufferEmpty");
                stream->frame_count += oframes;
                return stream->last_callback = paAbort;
            }
        }
    }

    if ( stream->rxbuff != NULL && stream->frame_count + frames_left > stream->offset ) {
        if ( stream->frame_count < stream->offset ) {
            offset = stream->offset - stream->frame_count;
            frames_left -= offset;
            in_data = (unsigned char *) in_data + offset*stream->rxbuff->elementSizeBytes;
        }

        iframes = PaUtil_WriteRingBuffer(stream->rxbuff, (const void *) in_data, frames_left);
        if ( iframes < frames_left && !stream->keep_alive ) {
            strcpy(stream->errorMsg, "ReceiveBufferFull");
            stream->frame_count += iframes;
            return stream->last_callback = paAbort;
        }
    }

    *stream->lastTime = *timeInfo;
    stream->frame_count += frame_count;
    return stream->last_callback;
}
