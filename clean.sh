#! /bin/bash

cd expCode/
# prevents from running on a past version
rm *.pyc

cd ../Stims/

# Matlab has been know to sometimes not overwrite wav files of the same name
rm 16000/*.wav
rm 24414/*.wav

