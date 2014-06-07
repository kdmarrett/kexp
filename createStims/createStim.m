% createStim.m
%   Author: Karl Marrett
%   Includes a preblock primer where target letters are played in their respective location and pitch, and
%   a post_block to provide time between trials.
% Key variables:
% WN number of wheels
% IWI timing between letters played from one wheel to the next
% ILI timing between letters played with one wheel (i.e. 'A' 'B')
% training blocks
% standardize variable names
% stimuli saved in block then trials
%TEST FOR SUBCOLUMN ALWAYS EQUALING ONE
%TEST FOR NO REPEATED LETTERS IN LETTER_TO_PITCH
%CREATE OTHER TESTS
%TRY/CATCH ASSERT CATCH EXCEPTION
% ALL ODDBALL AND TARGET TIMES RELATIVE TO STIM START
% 'R' NEEDS TO BE AT SEPARATE SPATIAL LOCATIONS
% Debug letterblock length
% Choose a smart IWI for 987

% DEFINE PATHS
PATH = '~/git/kexp/';%local letter and output directory
% PATH = '/Users/nancygrulke/git/kexp/'; %Mac
output_path = strcat(PATH, 'Stims/');%dir for all subject stimulus
letter_path = strcat(PATH, 'monotone_220Hz_24414'); %dir to untrimmed letters
K70_dir = strcat(PATH, 'K70'); % computed HRTF

% SET LETTERS
letterArray.alphabetic = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'}; %does not contain W
letterArray.displaced = {'A' 'B' 'F' 'O' 'E' 'M' 'I' 'T' 'J' 'C' 'H' 'Q' 'G' 'N' 'U' 'V' 'K' 'D' 'L' 'U' 'P' 'S' 'Z' 'R'}; %maximal phoneme separation no 'W' or 'Y'
subLetter = {'Z'}; %!! needs to be changed to non x[i] CV???

% General Parameters
fs = 24414; %letter sample rate
rms_amp = .01; %final normalization
letter_decibel = -10; %amplitude of letters in wheel; final stim normalized
odd_tone_decibel = -30; %amplitude of tone oddballs
odd_num_decibel = -30;
white_noise_decibel = -60;  %amplitude
noise = 0;  % bool adds noise (1)
distance_sound = 5; %distance for stimuli to be played in HRTF

%amplitude modulator shifts amplitude
AM_freq = [0 0 0 0 0 0 0 0]; %Hz rate of amplitude modulator elements for each wheel 0 for none
AM_pow =  [0 0 0 0 0 0 0 0]; %decibel of each AM for each corresponding wheel

% BOOLEANS FOR DESIGN FEATURES ORDERED: LETTERS PER WHEEL, ALPHABETIC VS. RANDOMLY SORTED, TARGET LETTER 'R' AS OPPOSED TO X[i],
% LETTER ORDERS ARE RETAINED ACROSS CYCLES, TONE IS ASSIGNED CONTIGUOUSLY AS OPPOSED TO RANDOMLY, 
condition_type = [0 0 0 0 0; 1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 1 0 1 0; 0 0 0 0 1];
trials_per_condition = 1;
condition_trials = repmat(trials_per_condition, length(condition_type));

%Global parameters of block design
scale_type = 'whole'; %string 'whole' or 'diatonic'
play_wheel = [1 1 1 1 1]; %boolean array to include certain wheels for training trials
tot_cyc = 10;
cycle_time = 3.500; % how long each wheel will last in seconds
postblock_sec = .5; %secs after letterblocks
postblock = ceil(postblock_sec * fs);  % convert to samples
preblock_prime_sec = 4.5; %secs to introduce primer letter
preblock = ceil(preblock_prime_sec * fs);
primer_start = 3000;  %sample # that primer letter will play in the preblock; must be less than preblock
extraSpace = 150000;   %methods to remove without throwing error ++++
total_letters = 24;
tone_constant = 0; %letters are assigned a random pitch and retained throughout
rearrange_cycles = 0;
enerMask = 0;
minTarg = 2;
maxTarg = 3;
target_time = [];

%% create specific vars for each condition
[m, n] = size(condition_type);
for y = 1:m; % repeats through each condition type
    block_name = strcat('block_', int2str(y));
    final_output_path = fullfile(output_path, block_name); % create dir for each block
    paradigm = condition_type(y, :);
    
    [wheel_matrix_info, possibleLetters, target_letter, rearrangeCycles, tone_constant] = assignParadigm(paradigm, letterArray);
    
    % COMPUTE MISC. BASIC PARAMS OF BLOCK
    for i = 1:length(wheel_matrix_info)
        ILI_sec(i) = cycle_time / wheel_matrix_info(i);
    end
    
    ILI = ceil(ILI_sec .* fs);
    wheel_token_Hz = wheel_matrix_info / cycle_time;
    
    if paradigm(1) % [9 8 7]
        [y, ind] = max(wheel_matrix_info);
        letterblock = ceil(cycle_time * tot_cyc + IWI * (ind - 1) + extraSpace); %only first wheel determines cycle length here when ILI > IWI
    else
        letterblock = ceil(cycle_time * tot_cyc + IWI * (length(wheel_matrix_info) - 1) + extraSpace); %rough sample length of each letterblock + extra space for last letter
        IWI = ceil(ILI(1) / length(wheel_matrix_info));
    end
    
    tot_sample = ceil(preblock + letterblock + postblock); %total samples in each wavfile ++++DEBUG?
    
    % GENERATE BLOCK FOR EACH CONDITION TYPE
    for z = 1:condition_trials(y); 
        targ_cyc = randi([minTarg maxTarg]); % no. target oddballs in each trial
        [ wheel_matrix, target_wheel_index, total_target_letters, droppedLetter ] = assignLetters( possibleLetters, wheel_matrix_info, target_letter, targ_cyc, tot_cyc, rearrangeCycles, enerMask, subLetter); % returns cell array of wheel_num elements
        [pitch_wheel, angle_wheel, total_pitches, list_of_pitches] = assignPitch(wheel_matrix_info, tot_cyc, scale_type); %returns corresponding cell arrays
        if tone_constant
            [ letter_to_pitch ] = assignConstantPitch( possibleLetters, total_letters, total_pitches, subLetter, droppedLetter );
        end
        final_sample = floor(zeros(tot_sample, 2)); % creates background track for each letter to be added on
        final_sample = final_sample + noise * (10^(white_noise_decibel / 20)) * randn(tot_sample, 2); %add white noise to background track
        primer_added = 0; %(re)sets whether primer has been added to each block
        wheel_sample_index = preblock; %delays each wheel by inter_wheel interval only refers to row for final_sample
        for j = 1:length(wheel_matrix_info)
            track_sample_index = 1; %used for each wheel for wheel_track indexing;
            if play_wheel(j)
            wheel_track = zeros(letterblock, 2); %(re)initialize background track for each wheel
            for k = 1:tot_cyc %for each cycle of the individual wheel
                for l = 1:wheel_matrix_info(j) %for each letter
                    letter = wheel_matrix{j}{k, l}; %finds the letter in wheel_matrix cell
                    if strcmp(letter, target_letter)
                        target_sample_index = wheel_sample_index + track_sample_index;
                        target_time = [target_time (target_sample_index / fs)];
                    end
                    if tone_constant
                        columnPitch = find(sum(strcmp(letter, letter_to_pitch)));
                        pitch_wheel{j}{k, l} = list_of_pitches{columnPitch};
                    end
                    pitch = pitch_wheel{j}{k, l}; %finds the pitch in pitch_wheel cell
                    angle = angle_wheel{j}(k, l); %finds the angle in angle_wheel for each letter
                    path = fullfile(letter_path, pitch, letter);
                    [letter_sound, fs] = wavread(path);
                    [L, R] = stimuliHRTF(letter_sound, fs, angle, distance_sound, K70_dir);
                    letter_sound_proc = (10 ^(letter_decibel/20)) * [L R];
                    foo = 1;
                    for m = track_sample_index: (track_sample_index + length(letter_sound_proc) - 1)%adds in superposition to final_sample at track_sample_index sample no
                        wheel_track(m, 1) = wheel_track(m, 1) + letter_sound_proc(foo, 1);
                        wheel_track(m, 2) = wheel_track(m, 2) + letter_sound_proc(foo, 2);
                        foo = foo + 1;
                    end
                    if ~primer_added %  primer not  added
                        if strcmp(letter, target_letter) %and is target letter
                            foo = 1;
                            for m = primer_start:(primer_start + length(letter_sound_proc) - 1)
                                final_sample(m, 1) = final_sample(m, 1) + letter_sound_proc(foo, 1); %add primer
                                final_sample(m, 2) = final_sample(m, 2) + letter_sound_proc(foo, 2);
                                foo = foo + 1;
                            end
                            primer_added = 1;
                        end
                    end % adding primer
                    track_sample_index = track_sample_index + ILI(j); %advances track_sample_index to the next letter slot intra wheel
                end %for each letter
            end %for each cycle
            [final_wheel] = gen_new_envelope(wheel_track, AM_freq(j), AM_pow(j), fs);
            [rows, cols] = size(final_wheel);
            if play_wheel(j) %add to final_sample only if specified by play_wheel
                foo = 1;
                for m = wheel_sample_index:(wheel_sample_index + rows - 1)
                    final_sample(m, 1) = final_sample(m, 1) + final_wheel(foo, 1); %adds each wheel in superposition to final_sample
                    final_sample(m, 2) = final_sample(m, 2) + final_wheel(foo, 2); %adds each wheel in superposition to final_sample
                    foo = foo + 1;
                end
            end
            
            end 
        wheel_sample_index = wheel_sample_index + IWI;
        end %for eachA wheel

        assert((targ_cyc == total_target_letters), 'Error in generate_letter_matrix_wheel creating correct number of targets')
        
        %stamp wav_name with each block labeled by paradigm condition
        wav_name = fullfile(final_output_path, strcat(int2str(z), '_3.5'))
        final_sample = rms_amp * final_sample / sqrt(mean(mean(final_sample.^2)));
        wavwrite(final_sample, fs, wav_name);
        %save('expVars') % for debugging purposes
    end
end




