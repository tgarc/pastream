#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"
#ifdef PYPA_DEBUG
#include <stdio.h>
#endif
#include <stddef.h>


void init_stream(Py_PaStream *stream)
{
    memset((void *) stream, 0, sizeof(Py_PaStream));
};

void reset_stream(Py_PaStream *stream) {
    memset((void *) &stream->lastTime, 0, sizeof(PaStreamCallbackTimeInfo));
    memset((void *) &stream->stats, 0, sizeof(Py_PaStreamStats));
};

int callback(
    void* in_data,
    void* out_data,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags status,
    void *user_data)
{
    ring_buffer_size_t frames_left = frame_count;
    ring_buffer_size_t oframes_left = frame_count;
    ring_buffer_size_t oframes, iframes, tempframes;
    Py_PaStream * stream = (Py_PaStream *) user_data;

    if ( status & 0xF ) {
        if ( status & paInputUnderflow )
            stream->stats.inputUnderflows++;
        if ( status & paInputOverflow )
            stream->stats.inputOverflows++;
        if ( status & paOutputUnderflow )
            stream->stats.outputOverflows++;
        if ( status & paOutputOverflow )
            stream->stats.outputOverflows++;
        stream->stats.xruns++;
    }

    if ( out_data != NULL && stream->txbuffer == NULL ) {
        memset((unsigned char *) out_data, 0, frame_count*stream->config.txElementSize);
    }
    else if ( stream->txbuffer != NULL ) {
        oframes = PaUtil_ReadRingBuffer(stream->txbuffer, out_data, oframes_left);

        // loop mode: this assumes that the buffer is only written before
        // stream is started (and never while its active), so we can safely
        // rewind when we hit the end
        while ( oframes < oframes_left && stream->loop ) {
            tempframes = stream->txbuffer->readIndex;

            // just in case frames == 0 or by some mistake the txbuffer is empty
            if ( tempframes == 0 ) break;

            PaUtil_FlushRingBuffer(stream->txbuffer);
            PaUtil_AdvanceRingBufferWriteIndex(stream->txbuffer, tempframes);
            oframes += PaUtil_ReadRingBuffer(stream->txbuffer,
                           (unsigned char *) out_data + oframes*stream->config.txElementSize,
                           oframes_left - oframes);
        }

        // No matter what happens, fill the remainder of the output buffer with zeros
        if ( oframes < frame_count ) {
            memset((unsigned char *) out_data + oframes*stream->config.txElementSize,
                   0,
                   (frame_count - oframes)*stream->config.txElementSize);
        }

        if ( oframes < frames_left ) {
            // strcpy(stream->errorMsg, "BufferEmpty");
            if (!stream->config.keepAlive) {
                stream->stats.frame_count += oframes;
                return paAbort;
            }
        }
    }

#ifdef PYPA_WIREMODE
    if ( out_data != NULL && in_data != NULL ) {
        memcpy(in_data, out_data, frame_count * stream->txElementSize);
    }
#endif

    if ( stream->rxbuffer != NULL ) {
        iframes = PaUtil_WriteRingBuffer(stream->rxbuffer, (const void *) in_data, frames_left);
        
        if ( iframes < frames_left ) {
            // strcpy(stream->errorMsg, "BufferFull");
            if (!stream->config.keepAlive) {
                stream->stats.frame_count += iframes;
                return paAbort;
            }
        }
    }

    stream->lastTime = *timeInfo;
    stream->stats.frame_count += frame_count;
    return paContinue;
}
