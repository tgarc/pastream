#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"


void init_stream(
    Py_PaBufferedStream *stream, 
    unsigned char allow_drops,
    long frames,
    long pad,
    unsigned long offset,
    PaUtilRingBuffer *rxbuff,
    PaUtilRingBuffer *txbuff) 
{
    stream->allow_drops = allow_drops;
    stream->frames = (frames >= 0 && pad >= 0) ? (frames + pad) : frames;
    // technically if frames < 0 padding has no meaning, but set it anyways to
    // give the user the least unexpected behavior
    stream->pad = pad;
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
    ring_buffer_size_t oframes, iframes;
    Py_PaBufferedStream *stream = (Py_PaBufferedStream *) user_data;
    long frames = stream->frames;
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
    if ( frames >= 0 && stream->frame_count + frames_left >= frames ) {
        frames_left = frames - stream->frame_count;
        stream->last_callback = paComplete;
    }

    if ( stream->txbuff != NULL ) {
        oframes = PaUtil_ReadRingBuffer(stream->txbuff, out_data, frames_left);

        // We're done reading frames! Or the writer was too slow; either way,
        // finish up by adding some zero padding.
        if ( oframes < frames_left ) {
            // Fill the remainder of the output buffer with zeros
            memset((unsigned char *) out_data + oframes*stream->txbuff->elementSizeBytes,
                   0, 
                   (frame_count - oframes)*stream->txbuff->elementSizeBytes);

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
            else if ( !stream->__autoframes && pad >= 0 ) {
                strcpy(stream->errorMsg, "BufferEmpty");
                stream->frame_count += oframes;
                return (stream->last_callback = paAbort);
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
        if ( iframes < frames_left && !stream->allow_drops ) {
            strcpy(stream->errorMsg, "BufferFull");
            stream->frame_count += iframes;
            return (stream->last_callback = paAbort);
        }
    }

    *stream->lastTime = *timeInfo;
    stream->frame_count += frame_count;
    return stream->last_callback;
}
