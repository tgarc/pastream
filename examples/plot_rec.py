#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import pastream as ps
import sys
import time
pi = np.pi


def draw(data):
    global plotdata
    xlim = list(ax.get_xlim())

    plotdata = np.roll(plotdata, -len(data), axis=0)
    plotdata[-len(data):] = data
    line.set_ydata(plotdata)

    return line,

with ps.BufferedInputStream(channels=1, device=sys.argv[1] if sys.argv[1:] else None) as stream:
    update = 1024
    plotdata = np.zeros(10*update, dtype=stream.dtype)

    fig, ax = plt.subplots()
    ax.grid()
    line, = ax.plot(plotdata, 'b-')
    ax.axis((0, len(plotdata), -1, 1))
    fig.tight_layout()
    inc = 0.9

    ani = animation.FuncAnimation(fig, draw, stream.chunks(update), blit=True, interval=0)
    plt.show()
