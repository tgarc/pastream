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
    unsigned long nframes = frame_count;
    ring_buffer_size_t oframes;
    Py_PsBufferedStream *stream = (Py_PsBufferedStream *) user_data;
    PaTime timedelta = timeInfo->currentTime - stream->lastTime;
    int returnCode = paContinue;

    stream->call_count += 1;

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
    if (stream->status&0xF)
        NULL;

    // (1) We've surpassed nframes: this is our last callback
    if (stream->nframes && stream->frame_count + nframes >= stream->nframes) {
        nframes = stream->nframes - stream->frame_count;
        returnCode = paComplete;
    }

    if (stream->duplexity & O_MODE) {
        oframes = PaUtil_ReadRingBuffer(stream->txq, out_data, nframes);

        // We're done reading frames! Or the writer was too slow; either way,
        // finish up by adding some zero padding.
        if (oframes < nframes) {
            // Fill the remainder of the output buffer with zeros
            memset(((unsigned char *) out_data) + oframes * stream->txq->elementSizeBytes,
                   0, 
                   (nframes - oframes) * stream->txq->elementSizeBytes);

            // (2) We don't need no stinkin' padding; we're done here
            if ( !(stream->nframes || stream->padframes) ) {
                returnCode = paComplete;
            } else if (!stream->nframes && stream->padframes) {
                // Figure out how much additional padding to insert and set nframes
                // equal to it
                stream->nframes = stream->frame_count + oframes + stream->padframes;
                // (3) We don't want to do an unncessary callback; end here
                if (stream->frame_count + nframes >= stream->nframes) {
                    nframes = stream->nframes - stream->frame_count;
                    returnCode = paComplete;
                }
            }
        }
    }

    if (stream->duplexity & I_MODE && stream->frame_count + nframes > stream->offset) {
        if (stream->frame_count < stream->offset) {
            nframes -= stream->offset - stream->frame_count;
        }

        if (PaUtil_WriteRingBuffer(stream->rxq, in_data, nframes) < nframes) {
            strcpy(stream->errorMsg, "Receive queue is full.");
            return paAbort;
        }
    }

    stream->lastTime = timeInfo->currentTime;
    stream->frame_count += frame_count;
    return returnCode;
}
