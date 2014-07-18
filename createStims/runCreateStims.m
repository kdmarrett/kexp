% createStim.m
%   Author: Karl Marrett
%   Includes a preblock primer where target letters are played in their respective location and pitch, and
%   a post_block to provide time between trials. Stimuli saved in trials  block then trials

% check that letter to pitch has no repeated elements
% fix file name from trimLetters
% does EM still work?
% male_trimmed 16000 
% Original 24414 convert in itunes
% isolet files at 16000
% for trim letters and trim instrNotes code in ways to stop if already been done?
% all wavrites need to scale to -1 to 1 but still keep on zero normalizeSoundVector email Ross
% make training doesnt work now
% shifted letter not working
% add in pause, start, space, delete
% now lots of zeros at the end of the file
% assign time vars needs to be completely revamped with final design and conditions in mind
% square away the final paradigms to clean up all the code
%  send sound samples with different speeds and different envelopes and add in the different conditoins that will be tested
% copy all non code to lester create git ignore for these files in git
% decide on final parameters and hard code while leaving past code commented in case last minute decisions
% get documentation for python like pydee in sublime
% email everyone 
% mrs0
% mnre0
% file saving in instrNotes needs to be revamped along with deleting all the old directories

close all
clear all
tic

all_cycle_time = [809.000]; % how long each wheel will last in seconds
for irunCreateStims = 1:length(all_cycle_time)

% DEFINE PATHS
PATH = '~/git/kexp/';%local letter and output directory
stimuli_path = strcat(PATH, 'Stims/');%dir for all subject stimulus
letter_path = strcat(PATH, 'Letters'); %dir to untrimmed letters
K70_dir = strcat(PATH, 'K70'); % computed HRTF
data_dir = strcat(PATH, 'data/');
instrNote_dir = strcat(PATH, 'instrNotes/'); % instrument notes

%GLOBAL INPUT PARAMETERS OF BLOCK DESIGN
rms_amp = .05; %final normalization (loudness)
letter_decibel = 10; %amplitude of letters in wheel; final stim normalized
white_noise_decibel = 0;  %amplitude
noise = 0;  % bool adds noise
distance_sound = 5; %distance for stimuli to be played in HRTF
scale_type = 'whole'; %string 'whole' or 'diatonic'
cycle_time = all_cycle_time(irunCreateStims);
tot_cyc = 5;
postblock_sec = 1.5; %secs after letterblocks
preblock_prime_sec = 4.5; %secs to introduce primer letter
primer_start = 3000;  %sample # that primer letter will play in the preblock; must be less than preblock
makeTraining = 0;
force_recreate = 1; %bool to force recreation of letters or pitches even if dir exists from previous run
instrument_dynamics = 'mf'; %mezzoforte 
env_instrNotes = 1; % bool for creating instrument notes based off of letter envelopes
start_sample_one = 1;
descend_pitch = [0 0 1];
speaker_list = {'mjc1', 'female_trimmed', 'mnre0'}; 
% SET LETTERS
% letterArray.alphabetic = {'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z'}; 
letterArray.alphabetic = {'Space', 'Pause', 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z', 'Read', 'Delete'}; 
letterArray.displaced =  {'Space', 'Pause', 'A' 'B' 'F' 'O' 'E' 'M' 'I' 'T' 'J' 'C' 'H' 'Q' 'G' 'N' 'U' 'V' 'K' 'D' 'L' 'U' 'P' 'S' 'Z' 'R' 'W' 'Y' 'Read' 'Delete'}; %maximal phoneme separation
assert(length(letterArray.alphabetic) == length(letterArray.displaced));
letter_samples = 10000; %length of each letter
total_letters = length(letterArray.alphabetic);
instr_list = {'Piano', 'Trumpet', 'Marimba'};
version_num = 1;
speaker_amp_weights = [1 1 1];
% speaker_amp_weights = [1 1.8 .5];
instr_amp_weights = [.5, .35, 1];

% ESTABLISH THE PITCH ORDERS FOR EACH WHEEL OF LETTERS
pitches.pent = {'0', '1.0', '2.0', '4.0', '5.0'};
pitches.diatonic = {'-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0' '-1.0','0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0'};
pitches.whole = {'-21.0', '-19.0', '-17.0', '-15.0', '-13.0', '-11.0', '-9.0', '-7.0', '-5.0', '-3.0', '-1.0', '1.0', '3.0', '5.0', '7.0', '9.0'};
pitches.all = {  '-21.0', '-20.0', '-19.0', '-18.0', '-17.0', '-16.0', '-15.0', '-14.0', '-13.0', '-12.0', '-11.0', '-10.0', '-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0' '-1.0','0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0' '9.0'};
% pitches.notes = {'C2' 'Db2' 'D2' 'Eb2' 'E2' 'F2' 'Gb2' 'G2' 'Ab2' 'A3' 'Bb3' 'B3' 'C3' 'Db3' 'D3' 'Eb3' 'E3' 'F3' 'Gb3'}; %encode by note name
pitches.notes = {'C3' 'Db3' 'D3' 'Eb3' 'E3' 'F3' 'Gb3' 'G3' 'Ab3' 'A3' 'Bb3' 'B3' 'C4' 'Db4' 'D4' 'Eb4' 'E4' 'F4' 'Gb4' 'G4' 'Ab4' 'A4' 'Bb4' 'B4' 'C5' 'Db5' 'D5' 'Eb5' 'E5' 'F5' 'Gb5'}; %encode by note name
pitches.notesWhole = {'C3', 'D3', 'E3', 'Gb3', 'Ab3', 'Bb3', 'C4', 'D4', 'E4', 'Gb4', 'Ab4', 'Bb4', 'C5'};
assert((length(pitches.notes) == length(pitches.all)), 'Error: note names do not cover range of possible pitches')


% BOOLEANS FOR DESIGN FEATURES, ORDERED: EACH WHEEL GROUP AT ORTHOGONAL FREQUENCY EM, EACH WHEEL GROUP GIVEN ON FREQUENCY AM , ALPHABETIC VS. MAXIMALLY DISPLACED, TARGET LETTER 'R' AS OPPOSED TO X[i],
% LETTER ORDERS ARE RETAINED ACROSS CYCLES, TONE IS ASSIGNED RANDOMLY AS OPPOSED TO CONTIGUOSLY, NO PITCH INFORMATION
condition_type = [0 0 0 0 0 0; 1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 1 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1 ];
[blocks, bar] = size(condition_type);
trials_per_condition = 1;
condition_trials = repmat(trials_per_condition, length(condition_type));
if makeTraining
    trials_per_training = 1;
    condition_trials_training = repmat(trials_per_condition,length(condition_type));
    reps = 2; %repeat again for training stimuli
else
    reps= 1;
end

%CREATE STIM FILE STRUCTURE
for i = 1:blocks
    fn = fullfile(stimuli_path, strcat('block_', int2str(i)));
    createStruct(fn);
end

% CREATE STIM TRAINING STRUCTURE
if makeTraining
    stimuli_path_train = fullfile(stimuli_path, 'training');
    for i = 1:blocks
        fn = fullfile(stimuli_path_train, strcat('block_', int2str(i)));
        createStruct(fn);
    end
end

% REPEATS THROUGH NON TRAINING THEN TRAINING TRIALS
for x = 1:reps  
    
    %% GENERATE BLOCK FOR EACH CONDITION TYPE
    [m, n] = size(condition_type);
    m = 1; % +++ only create the first 
    for y = 1:m; % repeats through each condition type
        
        % ASSIGN PARADIGM TO BLOCK
        block_name = strcat('block_', int2str(y));
        paradigm = condition_type(y, :);
        [wheel_matrix_info, possibleLetters, target_letter, rearrangeCycles, tone_constant, ener_mask, letters_used, token_rate_modulation,  AM_freq, AM_pow, shiftedLetters, instrNote_shifted, instrNote, envelope_type, letter_fine_structure, ILIms,  sync_cycles  ] = assignParadigm(paradigm, letterArray, env_instrNotes);
        [pitch_wheel, angle_wheel, total_pitches, list_of_pitches, start_semitone_index ] = assignPitch(wheel_matrix_info, tot_cyc, scale_type, pitches, descend_pitch);

        % TEST
        if ~(letters_used == total_letters)
            % fprintf('Error: not all letters ') % +++
        end

        % PREPARE LETTERS
        [fs, trim_letter_path, letterEnvelope, letterBits, mean_speaker_sample] = trimLetters(letter_samples, letter_path, letterArray, pitches, force_recreate, speaker_list, version_num, speaker_amp_weights, shiftedLetters, env_instrNotes);
        [nul] = trimInstrNotes(fs, instrNote_dir, letter_samples, pitches, instrument_dynamics, env_instrNotes, instr_list, speaker_list, letterEnvelope, list_of_pitches, force_recreate, letterArray, envelope_type, mean_speaker_sample, start_sample_one, start_semitone_index, wheel_matrix_info);

        % COMPUTE MISC. BASIC PARAMS OF BLOCK
        [ IWI, tot_trial, tot_wheel, letter_difference, min_wheel, preblock, ILI ] = assignTimeVars( wheel_matrix_info, fs, tot_cyc, letter_samples, token_rate_modulation, cycle_time, preblock_prime_sec, postblock_sec, ILIms );
        if tone_constant
            [ letter_to_pitch ] = assignConstantPitch( possibleLetters, total_letters, total_pitches);
        end

        %%  GENERATE EACH TRIAL WAV
        for z = 1:condition_trials(y);
            % target_time = []; % also clear target time from last trial
            [ wheel_matrix, target_wheel_index ] = assignLetters( possibleLetters, wheel_matrix_info, target_letter, tot_cyc, rearrangeCycles, ener_mask); % returns cell array of wheel_num elements
            if (x == 2) %if a training trial
                play_wheel = zeros(1,3); %bool array to include certain wheels for training trials
                play_wheel(target_wheel_index) = 1; % only include the target wheel
                output_path = stimuli_path_train;
            else
                play_wheel = [1 1 1]; 
                output_path = stimuli_path; 
            end
            final_output_path = fullfile(output_path, block_name); % create dir for each block

            final_sample = floor(zeros(tot_trial, 2)); % creates background track for each letter to be added on
            final_sample = final_sample + noise * (10^(white_noise_decibel / 20)) * randn(tot_trial, 2); %add white noise to background track
            primer_added = 0; %(re)sets whether primer has been added to each block
            wheel_sample_index = preblock; %delays each wheel by inter_wheel interval only refers to row for final_sample
            for j = 1:length(wheel_matrix_info) % for every wheel
                track_sample_index = 1; %used for each wheel for wheel_track indexing;
                if play_wheel(j)
                    current_speaker = speaker_list{j};
                    final_letter_path = fullfile(trim_letter_path, current_speaker);
                    wheel_track = zeros(tot_wheel(j), 2); %(re)initialize background track for each wheel
                    for k = 1:tot_cyc %for each cycle of the individual wheel
                        for l = 1:wheel_matrix_info(j) %for each letter
                            letter = wheel_matrix{j}{k, l}; %find letter in wheel_matrix cell

                            % FIND CONSTANT PITCH
                            if tone_constant
                                columnPitch = find(sum(strcmp(letter, letter_to_pitch)));
                                pitch_wheel{j}{k, l} = list_of_pitches{columnPitch};
                            end

                            % GET LETTER WAV FILE
                            pitch = pitch_wheel{j}{k, l}; %finds the pitch in pitch_wheel cell
                            angle = angle_wheel{j}(k, l); %finds the angle in angle_wheel for each letter
                            if shiftedLetters
                                path = fullfile(final_letter_path, pitch, letter);
                            else
                                path = fullfile(final_letter_path, letter);
                            end
                            if instrNote
                                note = semitoneToNote(pitch, pitches);
                                if instrNote_shifted
                                    fn = strcat(instr_list{j}, '.', instrument_dynamics, '.', note, '.wav');
                                else
                                    fn = strcat(instr_list{j}, '.', instrument_dynamics, '.', 'A4', '.wav');
                                end
                                if env_instrNotes
                                    trimInstrNotes_path = fullfile(instrNote_dir, 'trim', envelope_type, speaker_list{j}, instr_list{j}, pitch, letter, fn);
                                else
                                    trimInstrNotes_path = fullfile(instrNote_dir, 'trim', envelope_type, instr_list{j}, pitch, fn);
                                end
                                    instrNote_sound = wavread(trimInstrNotes_path);
                                instrNote_sound = instr_amp_weights(j) .* instrNote_sound;
                            else
                                instrNote_sound = zeros(letter_samples, 1);
                            end
                            [letter_sound, fs] = wavread(path);

                            % % VIEW
                            % plot(letter_sound)
                            % hold on
                            % plot(instrNote_sound, 'r')
                            % title(letter)
                            % waitforbuttonpress
                            % hold off

                            if letter_fine_structure
                                combined_sound = letter_sound + instrNote_sound;
                            else
                                combined_sound = instrNote_sound;
                            end
                            combined_sound = createGate(combined_sound, fs, 1,1);
                            [L, R] = stimuliHRTF(combined_sound, fs, angle, distance_sound, K70_dir);
                            % [L, R] = stimuliHRTF(letter_sound, fs, angle, distance_sound, K70_dir);
                            % [instrNote_L, instrNote_R] = stimuliHRTF(instrNote_sound, fs, angle, distance_sound, K70_dir);
                            % letter_sound_proc = (10 ^(letter_decibel / 20)) * [L R];
                            combined_sound_proc = (10 ^(letter_decibel / 20)) * [L R];
                            % instrNote_sound_proc = (10 ^(letter_decibel / 20)) * [instrNote_L, instrNote_R];

                            % % RECORD TARGET TIME
                            % if strcmp(letter, target_letter)
                            %     target_sample_index = wheel_sample_index + track_sample_index;
                            %     target_time = [target_time (target_sample_index / fs)];
                            %     if tone_constant %check that tone target tone is constant
                            %         if (length(target_time) > 2)
                            %             assert(strcmpi(pitch, old_pitch), 'Error: target tone not constant')
                            %         end
                            %         old_pitch = pitch;
                            %     end
                            % end
                            
                            % ADD LETTER TO WHEEL TRACK
                            local_index = track_sample_index;
                            for m = 1:(letter_samples - 1) % adds in superposition to final_sample at track_sample_index sample no
                                wheel_track(local_index, 1) = wheel_track(local_index, 1) + combined_sound_proc(m, 1);
                                wheel_track(local_index, 2) = wheel_track(local_index, 2) + combined_sound_proc(m, 2);
                                % if instrNote
                                %     wheel_track(local_index, 1) = wheel_track(local_index, 1) + instrNote_sound_proc(m, 1);
                                %     wheel_track(local_index, 2) = wheel_track(local_index, 2) + instrNote_sound_proc(m, 2);
                                % end
                                local_index = local_index + 1;
                            end
                            
                            % ADD THE PRIMER ONCE
                            if ~primer_added %  primer not  added
                                if strcmp(letter, target_letter) %and is target letter
                                    foo = 1; %always add primer at beginning of final track 
                                    for m = 1:(letter_samples - 1)
                                        final_sample(foo, 1) = final_sample(foo, 1) + combined_sound_proc(m, 1); %add primer
                                        final_sample(foo, 2) = final_sample(foo, 2) + combined_sound_proc(m, 2);
                                        foo = foo + 1;
                                    end
                                    primer_added = 1;
                                end
                            end % adding primer

                            % ADVANCE INDEX WHERE EACH LETTER WILL BE ADDED TO THE WHEEL TRACK BY ILI
                            track_sample_index = track_sample_index + ILI(j); 
                        end %for each letter
                        if sync_cycles
                            if ~token_rate_modulation
                                if (j == min_wheel)
                                    track_sample_index = track_sample_index + ILI(j) * letter_difference;
                                end
                            end
                        end
                    end %for each cycle
                    
                    % AMPLITUDE MODULATE EACH WHEEL SEPARATELY
                    [final_wheel] = createAMEnvelope(wheel_track, AM_freq(j), AM_pow(j), fs);
                    [final_wheel_rows(j), cols] = size(final_wheel);
                    assert((final_wheel_rows(j) == tot_wheel(j)), 'Error: check final track sample number calculation in assignTimeVars')

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
            wav_name = fullfile(final_output_path, strcat(int2str(z), '_', int2str(cycle_time * 1000), '_', speaker_list{1:length(speaker_list)}, '_', envelope_type, '_', 'ms', '.wav'));
            if exist(wav_name, 'file') %matlab does not overwrite on all systems
                delete(wav_name)
            end
            final_sample = rms_amp * (final_sample / sqrt(mean(mean(final_sample.^2))));
            % size(final_sample)
            % final_sample = normalizeSoundVector(final_sample);
            wavwrite(final_sample, fs, wav_name);
            % DEBUG HEAR +++
            % if y==1
            %     sound(final_sample, fs);
            % end
        end
    end
end
[done, fs] = wavread(fullfile(PATH, 'CLICKloud.WAV')); % to alert user
toc %print elapsed time
end
sound((.6 * done), fs);




