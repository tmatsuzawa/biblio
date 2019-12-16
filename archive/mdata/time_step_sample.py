# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:54:58 2015

@author: stephane
"""

import numpy as np
import os.path
import turbulence.display.graphes as graphes
import turbulence.cine as cine
import turbulence.tools.rw_data as rw_data


def main():
    file = '/Users/stephane/Documents/Experiences_local/Accelerated_grid/Set_time_scale/Logtime_movies/setup1_Dt_fps10000_n7000_Beta_1m.cine'
    times = get_cine_time(file)

    Dt = (times[1:] - times[:-1]) * 1000
    times = times[:-1]

    print(Dt)
    print(times)

    graphes.graphloglog(times, Dt, fignum=1, label='b^')
    graphes.legende('t (s)', 'Dt (ms)', '')

    compare_logtime(1)


def compare_logtime(fignum=1):
    Dt = 0.1
    alpha = 1.001;
    n = 7000;

    log_time_step(Dt, alpha, n)

    # read time step
    file = '/Users/stephane/Documents/Experiences_local/Accelerated_grid/Set_time_scale/H1180mm_S300mm_decay/param_NL_Dt.txt'

    t_s, Dt = read_time_step(file)
    graphes.graphloglog(t_s, Dt, fignum=fignum, label='ko')

    Dir = os.path.dirname(file)
    print(Dir)

    params = {'Dt': Dt, 'alpha': alpha, 'n': n}
    s = rw_data.gen_name(params)
    file_fig = Dir + '' + '/' + 'Dt_vs_t_data_mes1'  # +s

    print(file_fig)
    graphes.save_fig(0, file_fig)


def read_time_step(file):
    # read time step that was used to analyse the movie
    # every units should be in ms
    Header, lines = rw_data.read_dataFile(file, '\t', '\t')

    t_s = np.asarray(lines['Start'])
    Dt = np.asarray(lines['Dt'])

    t0 = t_s[0]
    t_s = (t_s - t0) / 1000  # t in s
    print(('t0 = ' + str(t0)))

    return t_s, Dt


def log_time_step(Dt, alpha, n):
    # Dt in ms
    # alpha : power coefficient , close to 1. greater than 1 means increasing time step
    # n : number of images
    Dt_list = [Dt * alpha ** i for i in range(n)]
    Dt_array = np.asarray(Dt_list)

    Dt_array = np.floor(Dt_array / Dt) * Dt

    #   graphes.graph(np.arange(1,n+1),Dt_array,label='r+')
    #   graphes.legende('# of image','Dt (ms)','')

    t = np.cumsum(Dt_array) / 1000  # t in s
    graphes.graphloglog(t, Dt_array, 1, label='r+')
    graphes.legende('t (s)', 'Dt (ms)', '')


def get_cine_time(file, display=False):
    c = cine.Cine(file)
    # something gets wrong with the computation of the cine length
    n = c.len()
    print(('Number of images : ' + str(n)))
    times = np.asarray([c.get_time(i) for i in range(n)])

    if display:
        graphes.graphloglog(times[1:], np.diff(times), label='k')
        graphes.legende('t (s)', 'Dt (s)', file)

    return times


if __name__ == '__main__':
    main()
    # file='/Volumes/labshared3/Stephane/Experiments/Accelerated_grid/2015_07_20/Log_movie/PIV_bv_hp_zoom_Dt05000fps_n14000_alpha500mu_X0mm_Zm150mm_fps5000_H1180mm_S300mm_1.cine'
    # get_cine_time(file,14000,True)
    # main()
