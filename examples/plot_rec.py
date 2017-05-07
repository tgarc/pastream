#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import pastream as ps
import sys
import time
pi = np.pi


update = 2048

delay = [0]*25

def draw(data):
    global plotdata, rmisses, maxdelay, counter, lasttime
    xlim = list(ax.get_xlim())

    delay[counter%len(delay)] = stream.rxbuff.read_available - len(data)
    counter += 1
    # print(time.time() - lasttime, len(data), stream.rxbuff.read_available - len(data), max(delay), stream._rmisses - rmisses)
    lasttime = time.time()
    rmisses = stream._rmisses

    plotdata = np.roll(plotdata, -len(data), axis=0)
    if not len(data): return line,
    plotdata[-len(data):] = data
    line.set_ydata(plotdata)

    return line,

try:               dev = int(sys.argv[1])
except IndexError: dev = None
except ValueError: dev = sys.argv[1]
    
lasttime = time.time()
counter = maxdelay = rmisses = 0
with ps.BufferedInputStream(channels=1, device=dev) as stream:
    plotdata = np.zeros(10*update, dtype=stream.dtype)

    fig, ax = plt.subplots()
    ax.grid()
    line, = ax.plot(plotdata, 'b-')
    ax.axis((0, len(plotdata), -1, 1))
    fig.tight_layout()
    inc = 0.9

    ani = animation.FuncAnimation(fig, draw, stream.chunks(), blit=True, interval=0, repeat=False)
    plt.show(block=True)
    
