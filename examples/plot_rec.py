#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import pastream as ps
import sys
import time
import traceback


__usage__ = "[device[ playback]]"

update = 1536

delay = [0]*25

def draw(data):
    global plotdata, rmisses, maxdelay, counter, lasttime
    xlim = list(ax.get_xlim())

    # delay[counter%len(delay)] = stream._rxbuffer.read_available - len(data)
    # counter += 1
    # print(time.time() - lasttime, len(data), stream.rxbuff.read_available - len(data), max(delay), stream._rmisses - rmisses)
    lasttime = time.time()
    rmisses = stream._rmisses

    plotdata = np.roll(plotdata, -len(data), axis=0)
    if not len(data): return line,
    plotdata[-len(data):] = data
    line.set_ydata(plotdata)

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

lasttime = time.time()
counter = maxdelay = rmisses = 0
with stream:
    plotdata = np.zeros(10*update, dtype=stream.dtype)

    fig, ax = plt.subplots()
    ax.grid()
    line, = ax.plot(plotdata, 'b-')
    ax.axis((0, len(plotdata), -1, 1))
    fig.tight_layout()
    inc = 0.9

    ani = animation.FuncAnimation(fig, draw, stream.chunks(playback=inpf, loop=True), blit=True, interval=0, repeat=False)
    try:
        plt.show(block=True)
    finally:
        print('xruns:', stream.xruns, 'misses:', stream._rmisses, stream._wmisses)
