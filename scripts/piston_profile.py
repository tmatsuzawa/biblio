
import argparse
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/')
import library.tools.process_data as process
import library.tools.handle_data as dhandle
import library.display.graph as graph
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-input', metavar='input', type=str, nargs='*', default=None, help='')
    parser.add_argument('-scale', dest='scale', type=float, default=1., help='')

    args = parser.parse_args()
    datalist = args.input
    scale = args.scale


    for datanum, datapath in enumerate(datalist):
        key, data, counter = dhandle.generate_data_dct(datapath, separation='\t')
        time, pos_com, vel_counts = data['var0'], data['var1'], data['var2']
        vel = [v*scale for v in vel_counts]
        acc = np.gradient(vel)
        indices = np.argwhere(np.abs(acc) > 0.01).flatten()
        vel_new = [vel[index] for index in indices]
        time_new = [time[index] for index in indices]

        graph.plot(time, vel, color='b',fignum=1)
        graph.scatter(time_new, vel_new, color='r', fignum=1, marker='x')
        graph.show()



