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
    int returnCode = paContinue;

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
        return paAbort;

    if (stream->nframes && stream->frame_count + frame_count > stream->nframes) {
        frame_count = stream->nframes - stream->frame_count;
        returnCode = paComplete;
    }

    if (stream->duplexity & O_MODE) {
        oframes = PaUtil_ReadRingBuffer(stream->txq, out_data, frame_count);
        if (oframes < frame_count) {
            // Fill the remainder of the output buffer with zeros
            memset((unsigned char *) out_data + oframes * stream->txq->elementSizeBytes,
                   0, (frame_count - oframes) * stream->txq->elementSizeBytes);

            // This is our last callback!
            if (!stream->nframes && !stream->padframes) {
                stream->frame_count += oframes;
                return paComplete;
            }
            else if (!stream->nframes && stream->padframes)
                stream->nframes = stream->frame_count + oframes + stream->padframes;

            if (stream->nframes > 0 && stream->frame_count + frame_count > stream->nframes) {
                frame_count = stream->nframes - stream->frame_count;
                returnCode = paComplete;
            }
        }
    }

    if (stream->duplexity & I_MODE && stream->frame_count + frame_count > stream->offset) {
        if (stream->offset > 0 && stream->frame_count < stream->offset) {
            frame_count -= stream->offset - stream->frame_count;
            stream->frame_count += stream->offset - stream->frame_count;
        }

        iframes = PaUtil_WriteRingBuffer(stream->rxq, in_data, frame_count);
        if (iframes < frame_count) {
            strcpy(stream->errorMsg, "Receive queue is full.");
            return paAbort;
        }
    }

    stream->lastTime = timeInfo->currentTime;
    stream->frame_count += frame_count;
    return returnCode;
}
