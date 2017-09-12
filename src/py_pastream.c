#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"
#include <math.h>
#ifdef PYPA_DEBUG
#include <stdio.h>
#endif


void init_stream(Py_PaStream *stream)
{
    *stream = Py_PaStream_default;
    reset_stream(stream);
};

void reset_stream(Py_PaStream *stream) {
    memset((void *) &stream->lastTime, 0, sizeof(PaStreamCallbackTimeInfo));

    stream->last_callback = paContinue;
    stream->status = 0;
    stream->frame_count = 0;
    stream->errorMsg[0] = '\0';
    stream->xruns = 0;
    stream->inputUnderflows = 0;
    stream->inputOverflows = 0;
    stream->outputUnderflows = 0;
    stream->outputOverflows = 0;
    if ( stream->__autoframes )
        stream->frames = -1;
    stream->__autoframes = 0;
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
    ring_buffer_size_t oframes_left = frame_count, oframes = 0, iframes;
    Py_PaStream *stream = (Py_PaStream *) user_data;
    long long frames = stream->frames;
    long pad = stream->pad;

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
    }

    // exit point (1 of 2)
    // We've surpassed frames: this is our last callback
    if ( frames >= 0 ) {
        if ( stream->frame_count + frames_left >= frames ) {
            frames_left = frames - stream->frame_count;
            stream->last_callback = paComplete;
        }
        if ( pad >= 0 && stream->frame_count + frames_left + pad >= frames ) {
            if ( abs(frames - pad) > stream->frame_count )
                oframes_left = abs(frames - pad) - stream->frame_count;
            else
                oframes_left = 0;
        }
    }

    if ( stream->txbuffer != NULL ) {
        if ( stream->__autoframes )
            oframes = 0;
        else
            oframes = PaUtil_ReadRingBuffer(stream->txbuffer, out_data, oframes_left);

        // We're done reading frames! Or the writer was too slow; either way,
        // finish up by adding some zero padding.
        if ( oframes < frames_left ) {
            // Fill the remainder of the output buffer with zeros
            memset((unsigned char *) out_data + oframes*stream->txbuffer->elementSizeBytes,
                   0,
                   (frame_count - oframes)*stream->txbuffer->elementSizeBytes);

            if ( frames < 0 ) {
                if ( pad >= 0 ) {
                    // Figure out how much additional padding to insert and set frames
                    // equal to it
                    stream->__autoframes = 1;
                    frames = stream->frames = stream->frame_count + oframes + pad;

                    // exit point (2 of 2)
                    // We don't want to do an unncessary callback; end here
                    if ( stream->frame_count + frames_left >= frames ) {
                        if ( stream->frame_count <= frames )
                            frames_left = frames - stream->frame_count;
                        else
                            frames_left = 0;
                        stream->last_callback = paComplete;
                    }
                }
                // else { pad indefinitely; }
            }
            else if ( !stream->__autoframes && pad >= 0 && oframes < oframes_left) {
                strcpy(stream->errorMsg, "BufferEmpty");
                stream->frame_count += oframes;
                return stream->last_callback = paAbort;
            }
        }
    }
    else if ( out_data != NULL ) {
        memset((unsigned char *) out_data, 0, frame_count*stream->txElementSize);
    }

    if ( stream->rxbuffer != NULL && stream->frame_count + frames_left > stream->offset ) {
        if ( stream->frame_count < stream->offset ) {
            offset = stream->offset - stream->frame_count;
            frames_left -= offset;
            in_data = (unsigned char *) in_data + offset*stream->rxbuffer->elementSizeBytes;
        }

        iframes = PaUtil_WriteRingBuffer(stream->rxbuffer, (const void *) in_data, frames_left);
        if ( iframes < frames_left ) {
            strcpy(stream->errorMsg, "BufferFull");
            stream->frame_count += iframes;
            return stream->last_callback = paAbort;
        }
    }

    stream->lastTime = *timeInfo;
    stream->frame_count += frame_count;
    return stream->last_callback;
}
