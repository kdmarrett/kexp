function [L,R] = stimuliHRTF(audio_in,fs,degree,distance, File_dir)

% KEMAR is positioned in the corner of the room 
% degree = KEMAR is positioned at (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
% distance = KEMAR is positioned at (1,2,3,4,5,6,7) corresponding to
% ( 20.0025   29.8450   44.4500   66.3575   98.9013  147.4787  219.9481 ) cm 


%File_dir = '~/Desktop/research/K70'; % linux home
%File_dir = '/home/labsner/Desktop/research/K70'; %linux lab
fileName = strcat('imp_k70_', num2str(abs(degree)),'_', num2str(distance),'.mat'); 
RevFile = fullfile(File_dir,fileName);
gain = 2^-24; % for some reason these are loud as hell

load(RevFile);
SFREQ = 44100; % sample rate of HRIR

% %set sample rate of input to the sample rate of HRIR
% if(fs ~= SFREQ)
%     x = resample(audio_in,SFREQ,fs);
% else
%     x = audio_in;
% end
x = audio_in;



info = load(fullfile(File_dir, 'startSampleInfo.mat'));
startSampleInd = find(strcmpi(fileName, info.fileName), 1);
if ~isempty(startSampleInd)
	startSample = info.startSample(startSampleInd);
else
	startSample = 1;
end
impl = impl(startSample:end);
impr = impr(startSample:end);

%set sample rate of HRIR to the sample rate of input
if(fs ~= SFREQ)
    impl = resample(impl,fs,SFREQ);
	impr = resample(impr,fs,SFREQ);
end

INlen = length(x); % length of token
IRlen = length(impl); % length of HRIR

% window the HRIR by cos^2 ramp window with ramp of 1 ms
RAMP_SAMPLES = floor(0.001*SFREQ);

WINlen = floor(5e-3*SFREQ); % the length of anechoic impulse response: first 5 ms

window = ones(WINlen,1);
window(WINlen-RAMP_SAMPLES:WINlen) = (cos(pi/(2*RAMP_SAMPLES)*(0:RAMP_SAMPLES)).^2)';
%window(RAMP_SAMPLES+1:-1:1) = (cos(pi/(2*RAMP_SAMPLES)*(0:RAMP_SAMPLES)).^2)'; % NO -- see below
window = [window; zeros(IRlen-WINlen,1)];

% DON'T WINDOW the start because that messes with the timing. startSample has been carefully determined.
lf = impl.*window;
rf = impr.*window;
lf = lf(1:WINlen);
rf = rf(1:WINlen);

%determine length of input + impulse response to nearest power of 2
power2 = WINlen+INlen-1;

%perform convolution by multiplication in the frequency domain and normalization
outL = real(ifft(fft(x,power2).*fft(lf,power2)));
outR = real(ifft(fft(x,power2).*fft(rf,power2)));
outL = outL(1:INlen);
outR = outR(1:INlen);

% conv(x,h) equivalent to filter(h, 1, x)
% where h is the impulse response and x is the signal

% normalization
% DON'T NORMALIZE -- RKM
% rmsL = sqrt(mean(outL.*outL));
% rmsR = sqrt(mean(outR.*outR));
% rms = max([rmsL,rmsR]);
% rmsM = sqrt(mean(audio_in.*audio_in));
% outL = outL./rms*rmsM;
% outR = outR./rms*rmsM;

% % resample to original sample rate
% L = resample(outL,fs,SFREQ)*gain;
% R = resample(outR,fs,SFREQ)*gain;
L = outL*gain;
R = outR*gain;

% swap left and right signals
if degree < 0
    temp = L;
    L = R;
    R = temp;
end
