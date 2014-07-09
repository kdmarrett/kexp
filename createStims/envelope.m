function [envelope] = envelope(signal)
% A quickly designed envelope detector 

hlowpass1 = dsp.FIRFilter(...
    'Numerator', firpm(20, [0 0.03 0.1 1], [1 1 0 0]));

DownsampleFactor = 1;
sigsq = 2*signal.*signal;
envelope = sqrt(step(hlowpass1, downsample(sigsq, DownsampleFactor)));


%downsample(abs(hilbert(letterSound{j}, fs_speaker)), 10);