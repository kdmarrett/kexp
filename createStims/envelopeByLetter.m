nfunction [envelope] = envelopeByLetter(signal, letter_samples, fs)
% A quickly designed envelope detector 

SPAN = 150;
iter = 2;
envelope = abs(hilbert(signal, letter_samples));
for i = 1:iter
	envelope = smooth(envelope, SPAN);
end

% wts = [1/24;repmat(1/500,11,1);1/24];
% envelope = conv(envelope,wts,'valid');
% x = 1:length(envelope);
% envelope = convhull(x', envelope);
% x = 1:length(envelope);
% [m, n] = size(envelope);
% xi = linspace(1, m, letter_samples);
% size(xi);
% y = envelope; 
% envelope = interp1(x, y, xi, 'spline', 'extrap');
% envelope = envelope';
% size(envelope);

% y = envelope;
% xi = 1:10000;
% envelope = interp1(x, y, xi);

% plot(x, y, 'b', xi, yi, 'r')
% envelope{j} = downsample(abs(hilbert(letterSound{j}, fs_speaker)), 10);

% hlowpass1 = dsp.FIRFilter(...
%     'Numerator', firpm(20, [0 0.03 0.1 1], [1 1 0 0]));

% DownsampleFactor = 1;
% sigsq = 2*signal.*signal;
% envelope = sqrt(step(hlowpass1, downsample(sigsq, DownsampleFactor)));
% assert(length(final_instrNote))

% splinefit could also work
