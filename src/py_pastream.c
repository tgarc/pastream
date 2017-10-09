#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"
#ifdef PYPA_DEBUG
#include <stdio.h>
#endif


unsigned int g_wiremode = 0;

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
    if ( stream->_autoframes && stream->pad >= 0 )
        stream->frames = -1;
    stream->_autoframes = 0;
};

int callback(
    const void* in_data,
    void* out_data,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags status,
    void *user_data)
{
    unsigned long offset = 0;
    ring_buffer_size_t frames_left = frame_count;
    ring_buffer_size_t oframes_left = frame_count;
    ring_buffer_size_t oframes, iframes, tempframes;
    Py_PaStream * stream = (Py_PaStream *) user_data;
    long long frames = stream->frames;
    long pad = stream->pad;

    // for testing only
    if ( g_wiremode )
        in_data = out_data;

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

    if ( frames >= 0 ) {
        // exit point (1 of 2)
        // We've surpassed frames: this is our last callback
        if ( stream->frame_count + frames_left >= frames ) {
            frames_left = frames - stream->frame_count;
            stream->last_callback = paComplete;
        }

        // Calcuate how many output frames are left to read (minus any padding)
        tempframes = frames - pad - stream->frame_count;
        if ( pad >= 0 && frames_left >= tempframes ) {
            oframes_left = tempframes > 0 ? tempframes : 0;
        }
    }

    if ( out_data != NULL && (stream->txbuffer == NULL || stream->_autoframes) ) {
        memset((unsigned char *) out_data, 0, frame_count*stream->txElementSize);
    }
    else if ( stream->txbuffer != NULL ) {
        oframes = PaUtil_ReadRingBuffer(stream->txbuffer, out_data, oframes_left);

        // loop mode: this assumes that the buffer is only written before
        // stream is started (and never while its active), so we can safely
        // rewind when we hit the end
        if ( stream->loop ) {
            while ( oframes < oframes_left ) {
                tempframes = stream->txbuffer->readIndex;

                // just in case frames == 0 or by some mistake the txbuffer is empty
                if ( tempframes == 0 ) break;

                PaUtil_FlushRingBuffer(stream->txbuffer);
                PaUtil_AdvanceRingBufferWriteIndex(stream->txbuffer, tempframes);
                oframes += PaUtil_ReadRingBuffer(stream->txbuffer, out_data, oframes_left);
            }
        }

        // No matter what happens, fill the remainder of the output buffer with zeros
        if ( oframes < frame_count )
            memset((unsigned char *) out_data + oframes*stream->txbuffer->elementSizeBytes,
                   0,
                   (frame_count - oframes)*stream->txbuffer->elementSizeBytes);

        if ( oframes < frames_left) {
            if ( frames < 0 ) {
                // Now that we've reached the end of buffer, calculate our
                // final frame count including padding and enter autoframes
                // mode
                if ( pad >= 0 ) {
                    stream->_autoframes = 1;
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
            }
            else if ( pad < 0 ) { // Just pad out to stream->frames samples
                stream->_autoframes = 1;
            }
            else if ( oframes < oframes_left ) {
                strcpy(stream->errorMsg, "BufferEmpty");
                stream->frame_count += oframes;
                return stream->last_callback = paAbort;
            }
        }
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
