#!/bin/sh
#echo "Installing cine module and copying cine2tiff script to ~/anaconda2/bin/“
#echo "(Your password is required to access this directory.)"

python setup.py install
sudo cp cine2tiff.py ~/anaconda2/bin/cine2tiff
sudo chmod a+rx ~/anaconda2/bin/cine2tiff

sudo cp cine2avi.py ~/anaconda2/bin/cine2avi
sudo chmod a+rx ~/anaconda2/bin/cine2avi

sudo cp img2avi.py ~/anaconda2/bin/img2avi
sudo chmod a+rx ~/anaconda2/bin/img2avi

sudo cp cine2sparse.py ~/anaconda2/bin/cine2sparse
sudo chmod a+rx ~/anaconda2/bin/cine2sparse

sudo cp make_s4d.py ~/anaconda2/bin/make_s4d
sudo chmod a+rx ~/anaconda2/bin/make_s4d

sudo cp multicine2avi.py ~/anaconda2/bin/multicine2avi
sudo chmod a+rx ~/anaconda2/bin/multicine2avi

sudo cp cine2mp4.py ~/anaconda2/bin/cine2mp4
sudo chmod a+rx ~/anaconda2/bin/cine2mp4
