#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import pastream as ps
import sys, os
import traceback


__usage__ = "[device[ playback]]"

update = 1536

window = np.hanning(update)

def draw(data):
    if len(data) < update: return line,
    line.set_ydata(np.abs(np.fft.rfft(data * window) / ((len(data)-1)/2)))
    return line,

def init():
    line.set_xdata(stream.samplerate*np.arange(update//2 + 1) / update)
    ax.axis((0, stream.samplerate / 2, 0, 1))
    fig.tight_layout()

    return line,

if '-h' in sys.argv or '--help' in sys.argv:
    print("usage:", __usage__)
    sys.exit(0)

try:               dev = int(sys.argv[1])
except IndexError: dev = None
except ValueError: dev = sys.argv[1]

try:               inpf = int(sys.argv[2])
except IndexError: inpf = None
except ValueError: inpf = sys.argv[2]

if inpf is not None:
    inpf = sf.SoundFile(inpf)
    stream = ps.DuplexStream.from_file(inpf, dev, channels=1)
else:
    stream = ps.InputStream(dev, channels=1)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', drawstyle='steps-mid')
ax.grid()

with stream:
    ani = animation.FuncAnimation(fig, draw,
                                  stream.chunks(update, update//2, playback=inpf, loop=True),
                                  blit=True, interval=0, repeat=False, init_func=init)
    plt.show(block=True)
