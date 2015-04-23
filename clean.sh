#! /bin/bash

# prevents from running on a past version
rm expCode/*.pyc

# clean up dummy trial data
rm -r Data/dum*

# Matlab sometimes does not overwrite wav files of the same name
# rm Stims/16000/*.wav
rm Stims/24414/*.wav

# remove all past trial var data (to recreate stimuli safely)
rm Data/Params/*.mat
