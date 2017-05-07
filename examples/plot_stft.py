#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import pastream as ps
import sys, os
import time

update = 2048

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

try:               dev = int(sys.argv[1])
except IndexError: dev = None
except ValueError: dev = sys.argv[1]

try:               inpf = int(sys.argv[2])
except IndexError: inpf = None
except ValueError: inpf = sys.argv[2]
    
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.grid()

# with ps.BufferedInputStream(channels=1, device=dev) as stream:
with ps.SoundFileStream(inpf=inpf, channels=1, device=dev) as stream:
    plotdata = np.zeros(update//2 + 1, dtype=float)
    ani = animation.FuncAnimation(fig, draw, stream.chunks(update, update//2),
                                  blit=True, interval=0, repeat=False, init_func=init)
    plt.show(block=True)
