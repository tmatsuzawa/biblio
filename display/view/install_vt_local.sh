#!/bin/bash

echo 'Installing view.py as a vt global command.'
echo 'Requires admin password to write to /usr/local/bin'

sudo ln -s $(pwd)/tangle_viewer.py /usr/local/bin/vt
sudo chmod a+x /usr/local/bin/vt

sudo ln -s $(pwd)/v4d_shader.py /usr/local/bin/v4
sudo chmod a+x /usr/local/bin/v4

sudo ln -s $(pwd)/movie.py /usr/local/bin/v4m
sudo chmod a+x /usr/local/bin/v4m
