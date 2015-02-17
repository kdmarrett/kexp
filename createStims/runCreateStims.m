% createStim.m
%   Author: Karl Marrett

close all
clear all
tic
	
% DEFINE PATHS
cd ..
PATH = cd ; %letter and output directory
cd createStims
stimuli_path = fullfile(PATH, 'Stims/');%dir for all subject stimulus
letter_path = fullfile(PATH, 'Letters', 'Files'); %dir to untrimmed letters
K70_dir = fullfile(PATH, 'K70'); % computed HRTF
instrNote_dir = fullfile(PATH, 'instrNotes/'); % instrument notes
lester_dir = '/Volumes/labdocs/kdmarrett/kexp';

session = input('Enter session number: ');
participant = input('Enter subject id: ', 's');
data_dir = fullfile(PATH, 'Data', 'Params', participant , int2str(session ));
stimuli_path = fullfile(stimuli_path, participant , int2str(session ));
createStruct(data_dir);
createStruct(stimuli_path);

%GLOBAL INPUT PARAMETERS OF BLOCK DESIGN
% Includes a preblock primer where target letters are played in their respective 
% location and pitch, and a post_block to provide time between trials.
instr_amp_weights = [2 1 4];
rms_amp = .05; %final normalization (loudness)
letter_decibel = 10; %amplitude of letters in wheel; final stim normalized
white_noise_decibel = 0;  %amplitude
noise = 0;  % bool adds noise
distance_sound = 5; %distance for stimuli to be played in HRTF
scale_type = 'whole'; %string 'whole' or 'diatonic'
tot_cyc = 9;
postblock_sec = 1.5; %secs after letterblocks
preblock_prime_sec = 	15.5; %seconds until letters start playing
primer_start = 3000;  %sample # that primer letter will play in preblock (less than preblock)
makeTraining = 0;
force_recreate = 1; %bool to force recreation of letters or pitches even if dir exists from previous run
default_fs = 16000;
instrument_dynamics = 'mf'; %mezzoforte
env_instrNotes = 0; % bool for creating instrument notes based off of letter envelopes
start_sample_one = 1; % start each instrument envelope at the begining of each letter regardless of letter power
descend_pitch = [0 0 1];
ILImsBase = 3 * 150;
ILIms = repmat(ILImsBase, 3, 1);
token_rates = [3 5 7];
English = 1; % English or German
wheel_matrix_info = [10 10 10];  %how many letters in each wheel

% SET LETTERS
if English
	% keep R letter in right wheel always
	letterArray.alphabetic = {'Space' 'Pause' 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' ...
	  'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' ...  % middle wheel
	  'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z' 'Read' 'Delete' };  % right wheel
	%maximal phoneme separation
	letterArray.displaced =  {'Space' 'Pause', 'A' 'B' 'F' 'O' 'E' 'M' 'I' 'T' ...
	'J' 'C' 'H' 'Q' 'G' 'N' 'U' 'V' 'K' 'D' ... % middle wheel
	'L' 'U' 'P' 'S' 'Z' 'R' 'W' 'Y' 'Read' 'Delete' }; % right wheel
	speaker_list = {'mjc1', 'female', 'mnre0'};
	speaker_amp_weights = [1 1 1];
else
	letterArray.alphabetic = {'Leer', 'Paus', 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H'...
	 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z',...
	  'Lesen', 'Losche'};
	letterArray.displaced =  {'Leer', 'Paus', 'A' 'B' 'F' 'O' 'E' 'M' 'I' 'T'...
	 'J' 'C' 'H' 'Q' 'G' 'N' 'U' 'V' 'K' 'D' 'L' 'U' 'P' 'S' 'Z' 'R' 'W' 'Y'...
	  'Lesen' 'Losche'}; %maximal phoneme separation
	speaker_list = {'male_1_G', 'female_1_G', 'male_2_G'}
	speaker_amp_weights = [1 1.5 1];
end
assert(length(letterArray.alphabetic) == length(letterArray.displaced));
letter_samples = 12000; %length of each letter
total_letters = length(letterArray.alphabetic);
instr_list = {'Piano', 'Trumpet', 'Marimba'};
version_num = 1;

% ESTABLISH THE PITCH ORDERS FOR EACH WHEEL OF LETTERS
pitches.pent = {'0', '1.0', '2.0', '4.0', '5.0'};
pitches.diatonic = {'-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0',...
 '-2.0' '-1.0','0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0'};
pitches.whole = {'-21.0', '-19.0', '-17.0', '-15.0', '-13.0', '-11.0', ...
'-9.0', '-7.0', '-5.0', '-3.0', '-1.0', '1.0', '3.0', '5.0', '7.0', '9.0'};
pitches.all = {  '-21.0', '-20.0', '-19.0', '-18.0', '-17.0', '-16.0', ...
'-15.0', '-14.0', '-13.0', '-12.0', '-11.0', '-10.0', '-9.0', '-8.0', ...
'-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0' '-1.0','0', '1.0', '2.0',...
 '3.0', '4.0', '5.0', '6.0', '7.0', '8.0' '9.0'};
% pitches.notes = {'C2' 'Db2' 'D2' 'Eb2' 'E2' 'F2' 'Gb2' 'G2' 'Ab2' 'A3' ...
% 'Bb3' 'B3' 'C3' 'Db3' 'D3' 'Eb3' 'E3' 'F3' 'Gb3'}; %encode by note name
pitches.notes = {'C3' 'Db3' 'D3' 'Eb3' 'E3' 'F3' 'Gb3' 'G3' 'Ab3' 'A3' 'Bb3' ...
'B3' 'C4' 'Db4' 'D4' 'Eb4' 'E4' 'F4' 'Gb4' 'G4' 'Ab4' 'A4' 'Bb4' 'B4' 'C5'... 
'Db5' 'D5' 'Eb5' 'E5' 'F5' 'Gb5'}; %encode by note name
pitches.notesWhole = {'C3', 'D3', 'E3', 'Gb3', 'Ab3', 'Bb3', 'C4', 'D4', ...
'E4', 'Gb4', 'Ab4', 'Bb4', 'C5'};
assert((length(pitches.notes) == length(pitches.all)),...
 'Error: note names do not cover range of possible pitches')

% condition_type = eye(7);
% condition_type = [zeros(1, 7); condition_type];
% condition_type(5, 2) = 1;
% smaller set of conditions for final testing
condition_type = [0, 0, 0, 0, 0, 0, 0; 0, 1, 0, 0, 0, 0, 0; 0, 1, 0, 1, 0, 0, 0];
% remove conditions where the letters are displaced for German
if ~English
	condition_type = condition_type([1:2, 4, 6:end], :)
end
[condition_no, bar] = size(condition_type);
trials_per_condition = 3;
condition_trials = repmat(trials_per_condition, length(condition_type), 1);

if makeTraining
	trials_per_training = 1;
	condition_trials_training = repmat(trials_per_condition,...
		length(condition_type));
	reps = 2; %repeat again for training stimuli
else
	reps= 1;
end

%CREATE STIM FILE STRUCTURE 
for i = 1:condition_no
	fn = fullfile(stimuli_path);
	createStruct(fn);
end

% CREATE STIM TRAINING STRUCTURE
if makeTraining
	stimuli_path_train = fullfile(stimuli_path, 'training');
	for i = 1:condition_no
		fn = fullfile(stimuli_path_train, strcat('block_', int2str(i)));
		createStruct(fn);
	end
end

% REPEATS THROUGH NON TRAINING THEN TRAINING TRIALS
for x = 1:reps
	
	%% GENERATE BLOCK FOR EACH CONDITION TYPE
	for y = 1:condition_no; % repeats through each condition type
		
		% ASSIGN PARADIGM TO BLOCK
		paradigm = condition_type(y, :);
		if x == 1
			condition_bin(y, :)  = reshape(dec2bin(paradigm)', [], 1)';
		end
		[possible_letters, target_letter, rearrangeCycles, tone_constant,...
		  ener_mask, letters_used, token_rate_modulation,  AM_freq, AM_pow,...
		  shiftedLetters, instrNote_shifted, instrNote, envelope_type,...
		   letter_fine_structure, sync_cycles  ] = assignParadigm(paradigm,...
		    letterArray, env_instrNotes, total_letters, wheel_matrix_info);
		[pitch_wheel, angle_wheel, total_pitches, list_of_pitches,...
		 start_semitone_index ] = assignPitch(wheel_matrix_info, tot_cyc, ...
		 	scale_type, pitches, descend_pitch );
		
		% PREPARE LETTERS
		[fs, trim_letter_path, letterEnvelope, mean_speaker_sample] = ...
		 trimLetters(total_letters, letter_samples, letter_path, letterArray,...
		 pitches, force_recreate, speaker_list, version_num, ...
		 speaker_amp_weights, shiftedLetters, env_instrNotes, English, ...
		 wheel_matrix_info, default_fs);
		if instrNote
			trimInstrNotes(fs, instrNote_dir, letter_samples, pitches, ...
				instrument_dynamics, env_instrNotes, instr_list, speaker_list,...
				 letterEnvelope, list_of_pitches, force_recreate, letterArray,...
				 envelope_type, mean_speaker_sample, start_sample_one, ...
				 start_semitone_index, wheel_matrix_info);
		end
		
		% COMPUTE MISC. BASIC PARAMS OF BLOCK
		[ IWI, tot_trial, tot_wheel, letter_difference, min_wheel, preblock, ...
		ILI, tot_wav_time, min_wheel_time, min_wheel_time_ind ] = ...
		assignTimeVars( wheel_matrix_info, fs, tot_cyc, letter_samples, ...
			token_rate_modulation, preblock_prime_sec, postblock_sec, ILIms, token_rates );
		if tone_constant
			[ letter_to_pitch ] = assignConstantPitch( possible_letters, ...
				total_letters, total_pitches, wheel_matrix_info, pitch_wheel);
		end
		
		%%  GENERATE EACH TRIAL WAV
		for z = 1:condition_trials(y);
			paradigm = condition_type(y, :);
			target_time = []; % also clear target time from last trial
			[ wheel_matrix, target_wheel_index ] = assignLetters( ...
			possible_letters, wheel_matrix_info, target_letter, tot_cyc, ...
			rearrangeCycles, ener_mask); % returns cell array of wheel_num elements
			if (x == 2) %if a training trial
				play_wheel = zeros(1,3); %bool array to include certain wheels for training trials
				play_wheel(target_wheel_index) = 1; % only include the target wheel
				output_path = stimuli_path_train;
			else
				play_wheel = [1 1 1];
				output_path = stimuli_path;
			end
			final_output_path = output_path;	
			
			% createTrial() start here
			% creates background track for each letter to be added on
			final_sample = floor(zeros(tot_trial, 2)); 
			final_sample = final_sample + noise * (10^(white_noise_decibel / ...
			 20)) * randn(tot_trial, 2); %add white noise to background track
			primer_added = 0; %(re)sets whether primer has been added to each block
			%delays each wheel by inter_wheel interval only refers to row for final_sample
			wheel_sample_index = preblock; 
			for j = 1:length(wheel_matrix_info) % for every wheel
				track_sample_index = 1; %used for each wheel for wheel_track indexing;
				if play_wheel(j)
					current_speaker = speaker_list{j};
					final_letter_path = fullfile(trim_letter_path, current_speaker);
					%(re)initialize background track for each wheel
					wheel_track = zeros(tot_wheel(j), 2); 
					for k = 1:tot_cyc %for each cycle of the individual wheel
						for l = 1:wheel_matrix_info(j) % for each letter
							letter = wheel_matrix{j}{k, l}; %find letter in wheel_matrix cell
							
							% FIND CONSTANT PITCH
							if tone_constant
								columnPitch = find(strcmp(letter, letter_to_pitch{j}));
								pitch = pitch_wheel{j}{1, columnPitch};
							else
								pitch = pitch_wheel{j}{k, l}; %finds the pitch in pitch_wheel cell
							end
							
							% GET LETTER WAV FILE
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
									note_path = fullfile(instrNote_dir, 'trim',...
									 envelope_type, speaker_list{j}, instr_list{j}, pitch, letter, fn);
								else
									note_path = fullfile(instrNote_dir, 'trim',...
									 envelope_type, instr_list{j}, pitch, strcat(note, '.wav' ));
								end
								instrNote_sound = wavread(note_path);
								instrNote_sound = instr_amp_weights(j) .* instrNote_sound;
							else
								instrNote_sound = zeros(letter_samples, 1);
							end
							[letter_sound, fs] = wavread(path);
							
							% VIEW
							% plot(letter_sound)
							% hold on
							% % plot(instrNote_sound, 'r')
							% title(letter)
							% waitforbuttonpress
							% hold off
							
							if letter_fine_structure
								combined_sound = letter_sound + instrNote_sound;
							else
								combined_sound = instrNote_sound;
							end
							combined_sound = createGate(combined_sound, fs, 1,1);
							[L, R] = stimuliHRTF(combined_sound, fs, angle, ...
								distance_sound, K70_dir);
							combined_sound_proc = (10 ^(letter_decibel / 20)) * [L R];
							
							% % RECORD TARGET TIME
							if strcmp(letter, target_letter)
							    target_sample_index = wheel_sample_index + ...
							     track_sample_index;
							    target_time = [target_time (target_sample_index / fs)];
							    if tone_constant %check that tone target tone is constant
							        if (length(target_time) > 2)
							            assert(strcmpi(pitch, old_pitch),...
							             'Error: target tone not constant')
							        end
							        old_pitch = pitch;
							    end
							end
							
							% ADD LETTER TO WHEEL TRACK
							local_index = track_sample_index;
							% adds in superposition to final_sample at track_sample_index sample no
							for m = 1:(letter_samples - 1) 
								wheel_track(local_index, 1) = ...
									wheel_track(local_index, 1) + combined_sound_proc(m, 1);
								wheel_track(local_index, 2) = ...
									wheel_track(local_index, 2) + combined_sound_proc(m, 2);
								local_index = local_index + 1;
							end
							
							% ADD THE PRIMER ONCE
							if ~primer_added %  primer not  added
								if strcmp(letter, target_letter) %and is target letter
									foo = 1; %always add primer at beginning of final track
									for m = 1:(letter_samples - 1)
										final_sample(foo, 1) = ...
											final_sample(foo, 1) + combined_sound_proc(m, 1); %add primer
										final_sample(foo, 2) = ...
											final_sample(foo, 2) + combined_sound_proc(m, 2);
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
					assert((final_wheel_rows(j) == tot_wheel(j)), ...
						'Error: check final track sample number calculation in assignTimeVars');

					% ADD EACH WHEEL TRACK TO FINAL SAMPLE TRACK
					local_index = wheel_sample_index;
					for m = 1:(final_wheel_rows(j) - 1)
						%adds each wheel in superposition to final_sample	
						final_sample(local_index, 1) = ...
							final_sample(local_index, 1) + final_wheel(m, 1); 
						final_sample(local_index, 2) = ...
							final_sample(local_index, 2) + final_wheel(m, 2);
						local_index = local_index + 1;
					end

				end

				% ADVANCE INDEX WHERE EACH WHEEL WILL BE ADDED TO FINAL TRACK BY IWI
				wheel_sample_index = wheel_sample_index + IWI;

			end %for each wheel

			% CREATETRIAL() END HERE

			%STAMP WAV_NAME WITH EACH BLOCK LABELED BY PARADIGM CONDITION
			% pass strings of binaries to Python for checking
			paradigm = dec2bin(paradigm)'; %cast to bin then to string
			paradigm_reshape = reshape(paradigm',[],1)';
			file_name = strcat( paradigm, '_', 'tr', int2str(z - 1))
			final_data_dir = fullfile(data_dir, file_name);
			save(final_data_dir, 'target_letter', 'target_time',...
			 'tot_wav_time', 'preblock_prime_sec', 'paradigm', ...
			 'possible_letters', 'preblock_prime_sec');
			wav_name = fullfile(final_output_path, strcat(file_name,'.wav'));
			% accounts for bug: matlab does not overwrite on all systems
			if exist(wav_name, 'file') 
				delete(wav_name)
			end
			final_sample = rms_amp * (final_sample / sqrt(mean(mean(final_sample.^2))));
			final_sample = normalizeSoundVector(final_sample);
			wavwrite(final_sample, fs, wav_name);   %  Stimuli saved by trial 
		end
	end
end
save( fullfile( data_dir, 'global_vars'), 'condition_bin', 'wheel_matrix_info',...
 'preblock_prime_sec', 'English', 'tot_cyc') % global variables for each subject and session
toc %print elapsed time
