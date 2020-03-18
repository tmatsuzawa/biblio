#!/bin/bash

echo 'Installing view.py as a vt global command.'
echo 'Requires admin password to write to /usr/bin'

sudo ln -s $(pwd)/tangle_viewer.py /usr/bin/vt
sudo chmod a+x /usr/bin/vt

sudo ln -s $(pwd)/v4d_shader.py /usr/bin/v4
sudo chmod a+x /usr/bin/v4

sudo ln -s $(pwd)/movie.py /usr/bin/v4m
sudo chmod a+x /usr/bin/v4m
