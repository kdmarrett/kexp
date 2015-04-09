clear all; close all;

% Script for changing all stimuli sampling frequency to a desired
% frequency for an experiment and for scaling sound to rms value

stim_rms = .01; %final normalization (loudness)
thresh = .01;
fs = 24414;
original_wav_dir = fullfile('../Stims', int2str(fs));
wavs = dir(fullfile(original_wav_dir, '*.wav'));
for i = 1:length(wavs)
    fprintf('Resampling wav file: %s', wavs(i).name);
    [final_sample, fs] = wavread(fullfile(original_wav_dir, ...
    wavs(i).name));
    [rsamp, csamp] = size(final_sample);
    sound_samps = []; % samples that are not zero for rms calc
    for samp = 1:rsamp
        if (abs(final_sample(samp, 1)) > thresh) 
            if (abs(final_sample(samp, 2)) > thresh) 
                sound_samps = [sound_samps; final_sample(samp, :)];
            end
        end
    end
    final_sample = stim_rms * (final_sample / ...
        mean(sqrt(mean(sound_samps.^2, 2))));
    wavname = strcat(original_wav_dir, wavs(i).name);
    wavwrite(final_sample, final_fs, wavname);
end

