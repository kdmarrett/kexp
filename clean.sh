#! /bin/bash

rm expCode/*.pyc
# prevents from running on a past version

rm -r Data/dummy*

# Matlab has been know to sometimes not overwrite wav files of the same name
rm Stims/16000/*.wav
rm Stims/24414/*.wav

