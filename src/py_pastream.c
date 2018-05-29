#include <portaudio.h>
#include <pa_ringbuffer.h>
#include "py_pastream.h"
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
    if ( stream->_autoframes && stream->pad >= 0 )
        stream->frames = -1;
    stream->_autoframes = 0;
};

int callback(
    void* in_data,
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
    int i, j, k;
    unsigned char temp, *ptemp;
    unsigned char samplesize;

    if ( status & 0xF ) {
        if ( status & paInputUnderflow )
            stream->inputUnderflows++;
        if ( status & paInputOverflow )
            stream->inputOverflows++;
        if ( status & paOutputUnderflow )
            stream->outputUnderflows++;
        if ( status & paOutputOverflow )
            stream->outputOverflows++;
        stream->xruns++;
        stream->status |= status;
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
        while ( oframes < oframes_left && stream->loop ) {
            tempframes = stream->txbuffer->readIndex;

            // just in case frames == 0 or by some mistake the txbuffer is empty
            if ( tempframes == 0 ) break;

            PaUtil_FlushRingBuffer(stream->txbuffer);
            PaUtil_AdvanceRingBufferWriteIndex(stream->txbuffer, tempframes);
            oframes += PaUtil_ReadRingBuffer(stream->txbuffer,
                           (unsigned char *) out_data + oframes*stream->txElementSize,
                           oframes_left - oframes);
        }

        // Re-map data to requested output channels
        if ( stream->txmapping != NULL ) {
            samplesize = stream->txElementSize / stream->txchannels;
            memcpy(stream->_mapping, stream->txmapping, stream->txchannels);

            for ( i = 0 ; i < stream->txchannels ; i++ ) {
                // inchannel = stream->_mapping[i];
                // outchannel = i + 1;
                if ( stream->_mapping[i] == i + 1 || stream->_mapping[i] == 0 )
                    continue;

                ptemp = (unsigned char *) out_data;
                if ( i + 1 == stream->txchannels || stream->_mapping[stream->_mapping[i] - 1] == 0 ) {
                    for ( j = 0 ; j < oframes ; j++, ptemp += stream->txElementSize ) {
                        memcpy(ptemp + i * samplesize, ptemp + (stream->_mapping[i] - 1)*samplesize, samplesize);
                    }
                }
                else {
                    for ( j = 0 ; j < oframes ; j++, ptemp += stream->txElementSize ) {
                        for ( k = 0 ; k < samplesize ; k++) {
                            temp = ptemp[i * samplesize + k];
                            ptemp[i * samplesize + k] = ptemp[(stream->_mapping[i] - 1)*samplesize + k];
                            ptemp[(stream->_mapping[i] - 1)*samplesize + k] = temp;
                        }
                    }
                }

                for ( k = i + 1 ; k < stream->txchannels ; k++) {
                    if (stream->_mapping[k] == i + 1)
                        stream->_mapping[k] = stream->_mapping[i];
                    else if (stream->_mapping[k] == stream->_mapping[i])
                        stream->_mapping[k] = i + 1;
                }
            }
            // Do channel zero out as a last step to avoid unnecessary swapping
            for ( i = 0 ; i < stream->txchannels ; i++ ) {
                if ( stream->_mapping[i] != 0 )
                    continue;

                ptemp = (unsigned char *) out_data + i * samplesize;
                for ( j = 0 ; j < oframes ; j++, ptemp += stream->txElementSize ) {
                    memset(ptemp, 0, samplesize);
                }
            }
        }

        // No matter what happens, fill the remainder of the output buffer with zeros
        if ( oframes < frame_count )
            memset((unsigned char *) out_data + oframes*stream->txElementSize,
                   0,
                   (frame_count - oframes)*stream->txElementSize);

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

#ifdef PYPA_WIREMODE
    if ( out_data != NULL && in_data != NULL ) {
        memcpy(in_data, out_data, frame_count * stream->txElementSize);
    }
#endif

    if ( stream->rxbuffer != NULL && stream->frame_count + frames_left > stream->offset ) {
        if ( stream->frame_count < stream->offset ) {
            offset = stream->offset - stream->frame_count;
            frames_left -= offset;
            in_data = (unsigned char *) in_data + offset*stream->rxElementSize;
        }

        if ( stream->rxmapping != NULL ) {
            samplesize = stream->rxElementSize / stream->rxchannels;
            memcpy(stream->_mapping, stream->rxmapping, stream->rxchannels);

            for ( i = 0 ; i < stream->rxchannels ; i++ ) {
                // inchannel = stream->_mapping[i];
                // outchannel = i + 1;
                if ( stream->_mapping[i] == i + 1 || stream->_mapping[i] == 0 )
                    continue;

                ptemp = (unsigned char *) in_data;
                if ( i + 1 == stream->rxchannels || stream->_mapping[stream->_mapping[i] - 1] == 0 ) {
                    for ( j = 0 ; j < frames_left ; j++, ptemp += stream->rxElementSize ) {
                        memcpy(ptemp + i * samplesize, ptemp + (stream->_mapping[i] - 1)*samplesize, samplesize);
                    }
                }
                else {
                    for ( j = 0 ; j < frames_left ; j++, ptemp += stream->rxElementSize ) {
                        for ( k = 0 ; k < samplesize ; k++) {
                            temp = ptemp[i * samplesize + k];
                            ptemp[i * samplesize + k] = ptemp[(stream->_mapping[i] - 1)*samplesize + k];
                            ptemp[(stream->_mapping[i] - 1)*samplesize + k] = temp;
                        }
                    }
                }

                for ( k = i + 1 ; k < stream->rxchannels ; k++) {
                    if (stream->_mapping[k] == i + 1)
                        stream->_mapping[k] = stream->_mapping[i];
                    else if (stream->_mapping[k] == stream->_mapping[i])
                        stream->_mapping[k] = i + 1;
                }
            }
            // Do channel zero out as a last step to avoid unnecessary swapping
            for ( i = 0 ; i < stream->rxchannels ; i++ ) {
                if ( stream->_mapping[i] != 0 )
                    continue;

                ptemp = (unsigned char *) in_data + i * samplesize;
                for ( j = 0 ; j < frames_left ; j++, ptemp += stream->rxElementSize ) {
                    memset(ptemp, 0, samplesize);
                }
            }
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
