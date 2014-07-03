% createStim.m
%   Author: Karl Marrett
%   Includes a preblock primer where target letters are played in their respective location and pitch, and
%   a post_block to provide time between trials. Stimuli saved in trials  block then trials
% TEST FOR NO REPEATED LETTERS IN LETTER_TO_PITCH
% ALL ODDBALL AND TARGET TIMES RELATIVE TO STIM START
% 'R' NEEDS TO BE AT SEPARATE SPATIAL LOCATIONS or choose other unique chars
% block 2 error
% choose subletter needs to be changed to non x[i] CV???
% need to envelope trimmed sounds
% finally check all conditions

tic
% DEFINE PATHS
PATH = '~/git/kexp/';%local letter and output directory
stimuli_path = strcat(PATH, 'Stims/');%dir for all subject stimulus
unprocessed_letter_path = strcat(PATH, 'monotone_220Hz_24414'); %dir to untrimmed letters
K70_dir = strcat(PATH, 'K70'); % computed HRTF
data_dir = strcat(PATH, 'data/');

% SET LETTERS
letterArray.alphabetic = {'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z'}; 
letterArray.displaced =  {'A' 'B' 'F' 'O' 'E' 'M' 'I' 'T' 'J' 'C' 'H' 'Q' 'G' 'N' 'U' 'V' 'K' 'D' 'L' 'U' 'P' 'S' 'Z' 'R' 'W' 'Y'}; %maximal phoneme separation
subLetter = {'Z'}; 
letter_samples = 10000; %length of each letter

% ESTABLISH THE PITCH ORDERS FOR EACH WHEEL OF LETTERS
pitches.pent = {'0', '1.0', '2.0', '4.0', '5.0'};
pitches.diatonic = {'-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0' '-1.0','0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0'};
pitches.whole = {'-9.0', '-7.0', '-5.0', '-3.0', '-1.0', '1.0', '3.0', '5.0', '7.0', '9.0'};   
pitches.all = {'-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0' '-1.0','0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0' '9.0'};

% PREPARE LETTERS
recreate_trimmed_letters = 1; %bool recreates trimmed letters even if dir exists
[fs, final_letter_path] = trimLetters(letter_samples, unprocessed_letter_path, letterArray, pitches, recreate_trimmed_letters);

% GENERAL PARAMETERS
rms_amp = 5; %final normalization
letter_decibel = 10; %amplitude of letters in wheel; final stim normalized
% odd_tone_decibel = -30; %amplitude of tone oddballs
% odd_num_decibel = -30;
white_noise_decibel = 0;  %amplitude
noise = 0;  % bool adds noise
distance_sound = 5; %distance for stimuli to be played in HRTF

%AMPLITUDE MODULATOR SHIFTS AMPLITUDE
AM_freq = [0 0 0 0 0 0 0 0]; %Hz rate of amplitude modulator elements for each wheel 0 for none
AM_pow =  [0 0 0 0 0 0 0 0]; %decibel of each AM for each corresponding wheel

% BOOLEANS FOR DESIGN FEATURES, ORDERED: LETTERS PER WHEEL, ALPHABETIC VS. MAXIMALLY DISPLACED, TARGET LETTER 'R' AS OPPOSED TO X[i],
% LETTER ORDERS ARE RETAINED ACROSS CYCLES, TONE IS ASSIGNED CONTIGUOUSLY AS OPPOSED TO RANDOMLY, ENERGETIC VS. INFO MASK
condition_type = [0 0 0 0 0 0; 1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 1 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1 ];
[blocks, bar] = size(condition_type);
trials_per_condition = 1;
condition_trials = repmat(trials_per_condition, length(condition_type));
reps = 1;
makeTraining = 0;
if makeTraining
    trials_per_training = 1;
    condition_trials_training = repmat(trials_per_condition,length(condition_type));
    reps = 2;
end

%CREATE STIM FILE STRUCTURE
for i = 1:blocks
    fn = fullfile(stimuli_path, strcat('block_', int2str(i)));
    createStruct(fn);
end
% Stim training structure
if makeTraining
    stimuli_path_train = fullfile(stimuli_path, 'training');
    for i = 1:blocks
        fn = fullfile(stimuli_path_train, strcat('block_', int2str(i)));
        createStruct(fn);
    end
end

%GLOBAL PARAMETERS OF BLOCK DESIGN
scale_type = 'whole'; %string 'whole' or 'diatonic'
tot_cyc = 10;
cycle_time = 2.000; % how long each wheel will last in seconds
cycle_sample = ceil(cycle_time * fs);
postblock_sec = 1.5; %secs after letterblocks
postblock = ceil(postblock_sec * fs);  % convert to samples
preblock_prime_sec = 4.5; %secs to introduce primer letter
preblock = ceil(preblock_prime_sec * fs);
primer_start = 3000;  %sample # that primer letter will play in the preblock; must be less than preblock
total_letters = length(letterArray.alphabetic);
rearrange_cycles = 0;
minTarg = 2;
maxTarg = 3;
target_time = [];

% REPEATS THROUGH NON TRAINING THEN TRAINING TRIALS  
for x = 1:reps  
    if (x == 2) %if a training trial
        play_wheel = zeros(1,3); %bool array to include certain wheels for training trials
        play_wheel(target_wheel_index) = 1; % only include the target letter
        output_path = stimuli_path_train;
    else
        play_wheel = [1 1 1]; 
        output_path = stimuli_path; 
    end

    %% GENERATE BLOCK FOR EACH CONDITION TYPE
    [m, n] = size(condition_type);
    for y = 1:m; % repeats through each condition type
        % if (y == 1) ++++++
        block_name = strcat('block_', int2str(y));
        final_output_path = fullfile(output_path, block_name); % create dir for each block
        paradigm = condition_type(y, :);
        [wheel_matrix_info, possibleLetters, target_letter, rearrangeCycles, tone_constant, ener_mask, letters_used, token_rate_modulation] = assignParadigm(paradigm, letterArray);
        assert((letters_used == total_letters), 'Error: not all letters assigned') 
        
        % COMPUTE MISC. BASIC PARAMS OF BLOCK
        if tone_constant
            [ letter_to_pitch ] = assignConstantPitch( possibleLetters, total_letters, total_pitches, subLetter, droppedLetter );
        end
        [ ILI, IWI, tot_trial, tot_wheel ] = assignTimeVars(wheel_matrix_info, cycle_sample, fs, tot_cyc, letter_samples, token_rate_modulation, preblock, postblock );

        if (y ==2)  %+++++++
            play_wheel
        end
        
        %%  GENERATE EACH TRIAL WAV
        for z = 1:condition_trials(y);
            targ_cyc = randi([minTarg maxTarg]); % no. target oddballs in each trial
            [ wheel_matrix, target_wheel_index, droppedLetter ] = assignLetters( possibleLetters, wheel_matrix_info, target_letter, targ_cyc, tot_cyc, rearrangeCycles, ener_mask, subLetter); % returns cell array of wheel_num elements
            [pitch_wheel, angle_wheel, total_pitches, list_of_pitches] = assignPitch(wheel_matrix_info, tot_cyc, scale_type, pitches); %returns corresponding cell arrays
            
            final_sample = floor(zeros(tot_trial, 2)); % creates background track for each letter to be added on
            final_sample = final_sample + noise * (10^(white_noise_decibel / 20)) * randn(tot_trial, 2); %add white noise to background track
            primer_added = 0; %(re)sets whether primer has been added to each block
            wheel_sample_index = preblock; %delays each wheel by inter_wheel interval only refers to row for final_sample
            for j = 1:length(wheel_matrix_info) % for every wheel
                track_sample_index = 1; %used for each wheel for wheel_track indexing;
                if play_wheel(j)
                    wheel_track = zeros(tot_trial(j), 2); %(re)initialize background track for each wheel
                    for k = 1:tot_cyc %for each cycle of the individual wheel
                        for l = 1:wheel_matrix_info(j) %for each letter
                            letter = wheel_matrix{j}{k, l}; %find letter in wheel_matrix cell
                            
                            % RECORD TARGET TIME
                            if strcmp(letter, target_letter)
                                target_sample_index = wheel_sample_index + track_sample_index;
                                target_time = [target_time (target_sample_index / fs)];
                            end

                            % FIND CONSTANT PITCH
                            if tone_constant
                                columnPitch = find(sum(strcmp(letter, letter_to_pitch)));
                                pitch_wheel{j}{k, l} = list_of_pitches{columnPitch};
                            end

                            % GET LETTER WAV FILE
                            pitch = pitch_wheel{j}{k, l}; %finds the pitch in pitch_wheel cell
                            angle = angle_wheel{j}(k, l); %finds the angle in angle_wheel for each letter
                            path = fullfile(final_letter_path, pitch, letter);
                            [letter_sound, fs] = wavread(path);
                            [L, R] = stimuliHRTF(letter_sound, fs, angle, distance_sound, K70_dir);
                            letter_sound_proc = (10 ^(letter_decibel / 20)) * [L R];
                            
                            % ADD LETTER TO WHEEL TRACK
                            local_index = track_sample_index;
                            for m = 1:(letter_samples - 1) % adds in superposition to final_sample at track_sample_index sample no
                                wheel_track(local_index, 1) = wheel_track(local_index, 1) + letter_sound_proc(m, 1);
                                wheel_track(local_index, 2) = wheel_track(local_index, 2) + letter_sound_proc(m, 2);
                                local_index = local_index + 1;
                            end
                            
                            % ADD THE PRIMER ONCE
                            if ~primer_added %  primer not  added
                                if strcmp(letter, target_letter) %and is target letter
                                    % size(letter_samples)
                                    % letter_samples
                                    % fprintf('c')
                                    foo = 1; %always add primer at beginning of final track 
                                    for m = 1:(letter_samples - 1)
                                        final_sample(foo, 1) = final_sample(foo, 1) + letter_sound_proc(m, 1); %add primer
                                        final_sample(foo, 2) = final_sample(foo, 2) + letter_sound_proc(m, 2);
                                        foo = foo + 1;
                                    end
                                    primer_added = 1;
                                end
                            end % adding primer

                            % ADVANCE INDEX WHERE EACH LETTER WILL BE ADDED TO THE WHEEL TRACK BY ILI
                            track_sample_index = track_sample_index + ILI(j); 
                        end %for each letter
                    end %for each cycle
                    
                    % AMPLITUDE MODULATE EACH WHEEL SEPARATELY
                    [final_wheel] = createEnvelope(wheel_track, AM_freq(j), AM_pow(j), fs);
                    [final_wheel_rows(j), cols] = size(final_wheel);
                    final_wheel_rows
                    tot_wheel
                    assert(final_wheel_rows == tot_wheel(j), 'Error: check final track sample number calculation in assignTimeVars')

                    % ADD EACH WHEEL TRACK TO FINAL SAMPLE TRACK
                    local_index = wheel_sample_index;
                    for m = 1:(final_wheel_rows(j) - 1)
                        final_sample(local_index, 1) = final_sample(local_index, 1) + final_wheel(m, 1); %adds each wheel in superposition to final_sample
                        final_sample(local_index, 2) = final_sample(local_index, 2) + final_wheel(m, 2); 
                        local_index = local_index + 1;
                    end

                end 
                % ADVANCE INDEX WHERE EACH WHEEL WILL BE ADDED TO FINAL TRACK BY IWI
                wheel_sample_index = wheel_sample_index + IWI;
            end %for each wheel
                    
            %STAMP WAV_NAME WITH EACH BLOCK LABELED BY PARADIGM CONDITION
            filename = strcat(data_dir, block_name, '_t_', int2str(z));
            wav_name = fullfile(final_output_path, strcat(int2str(z), '_', int2str(cycle_time * 1000), strcat('ms', '.wav')));
            final_sample = rms_amp * (final_sample / sqrt(mean(mean(final_sample.^2))));  
            wavwrite(final_sample, fs, wav_name);
        end
    % end % +++
    end
end
assert((length(target_time) == targ_cyc), 'Error in target_time: not a time stamp for every target')
[done, fs] = wavread(fullfile(PATH, 'CLICKloud.WAV')); % to alert user
target_time
toc %print elapsed time
sound(done, fs);




