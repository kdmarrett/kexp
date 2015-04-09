function [ trim_letter_path, letterEnvelope, mean_speaker_sample ] = ...
	trimLetters(total_letters, letter_samples, letter_path, letterArray, ...
		pitches, force_recreate, speaker_list, version_num, ...
		speaker_amp_weights, shiftedLetters, env_instrNotes, English, ...
		wheel_matrix_info, default_fs, new_fs);
% Writes all new letters with the specified sample length of letter_samples and saves into 
% folder trim_letter_path a subdirectory of letter_path

fs = [];
letterEnvelope = {};    
summed_letter_speaker = zeros(letter_samples, length(speaker_list));
mean_speaker_sample = zeros(3, 1);
index = 1;
index2 = 1;
for x = 1:length(speaker_list)
	fs_speaker = default_fs; % default letter sample rate
	% if letters are shifted update the path
	if shiftedLetters
		input_letter_path = fullfile(letter_path, 'shiftedLetters', speaker_list{x});
		assert(exist(input_letter_path, 'dir'), 'Error: you must shift the letters using praat first before using this speaker')
	else
		input_letter_path = fullfile(letter_path, 'rawLetters', strcat(speaker_list{x}, '_manuallyTrim'));
	end
	% create final output directory for trimmed letters
	trim_letter_path = fullfile(letter_path, 'finalShiftTrimLetters', int2str(letter_samples)); % needs to also include the speaker folders for force_recreate to work properly
	output_path = fullfile(trim_letter_path, speaker_list{x});
	% force recreate forces all raw letters to be retrimmed and saved
	if ((~exist(trim_letter_path, 'dir')) || force_recreate)
		letterSound = {};
		trimmedLetters = {};
		% if letters are shifted loop through all possible semitones
		if shiftedLetters
			iterations = length(pitches.all);
		else
			iterations = 1;
		end
		for i = 1:iterations 
			%loop through all letter in wheel in each semitone dir
			for j = 1:total_letters
			% for j = 1:wheel_matrix_info(x) %implement for tones and efficiency
				if shiftedLetters
					fp = fullfile(input_letter_path, pitches.all{i});
				else
					fp = input_letter_path;
				end
				if English
					temp_fn = strcat(speaker_list{x}, '-', letterArray.alphabetic{j}, int2str(version_num), '-t', '.wav');
					if (strcmpi(speaker_list{x}, 'Original') || strcmpi(speaker_list{x}, 'male_trimmed') || strcmpi(speaker_list{x}, 'female'))
						fn = strcat(letterArray.alphabetic{j});
					elseif exist(fullfile(fp, temp_fn), 'file')
						fn = temp_fn;
					else
						fn = strcat(speaker_list{x}, '_', letterArray.alphabetic{j}, int2str(version_num), '.wav');
					end
					if (strcmpi(letterArray.alphabetic(j), 'Read') || ... 
						strcmpi(letterArray.alphabetic(j), 'Space') || ...
						strcmpi(letterArray.alphabetic(j), 'Delete') || ...
						strcmpi(letterArray.alphabetic(j), 'Pause'))
						ff = fullfile(letter_path, 'rawLetters', ...
							'kdm_manuallyTrim', strcat(letterArray.alphabetic{j}, '.wav'));
					else
						ff = fullfile(fp, fn); 
					end
				else	
					ff = fullfile(fp, strcat(letterArray.alphabetic{j}, '.wav'));
				end
				try
					[letterSound{j}, fs_speaker, letterBits] = wavread(ff);  % letter wavs for each semitone
				catch	% this hack gets around wav file types saved in compressed formats	
					f=fopen(ff,'r+'); fseek(f,20,0); fwrite(f,[3 0]); fclose(f);  
					[letterSound{j}, fs_speaker, letterBits] = wavread(ff);
				end
				if env_instrNotes
					letterEnvelope{j} = envelopeByLetter(letterSound{j}, letter_samples, fs_speaker); 
					% VISUALIZE:
					% plot(letterEnvelope{j})
					% hold on
					% plot(letterEnvelope{j}, 'r')
					% title(letterArray.alphabetic{j})
					% waitforbuttonpress
					% hold off
				end

				% CHANGE OVERALL AMPLITUDE OF INDIVIDUAL SPEAKERS
				letterSound{j} = speaker_amp_weights(x) .* letterSound{j};
			end

			% TRIM EACH LETTER 
			for j = 1:total_letters
				% adjust output path by pitch 
				if shiftedLetters
					final_output_path = fullfile(output_path, pitches.all{i});
				else
					final_output_path = output_path;
				end
				createStruct(final_output_path);
				% clean final sound
				trimmedLetters{j} = trimSoundVector(letterSound{j}, fs_speaker, letter_samples, 1, 1);
				normalLetters{j} = createGate(trimmedLetters{j}, fs_speaker, 1, 1);
				final_trimmedLetters{j} = normalizeSoundVector(normalLetters{j});
				% estimate mean power of all letters for deciding instrument note timings
				pwr_est = final_trimmedLetters{j}.^2;
				summed_letter_speaker(:, x) = summed_letter_speaker(:, x) + pwr_est;
				fname = fullfile(final_output_path, strcat(letterArray.alphabetic{j}, '.wav'));
                [P, Q] = rat(new_fs, fs_speaker);
                final_trimmedLetters{j}= resample(final_trimmedLetters{j}, P, Q);
				wavwrite(final_trimmedLetters{j}, new_fs, fname);
			end    
		end
	end
	fs(x) = fs_speaker;

	% DETERMINE MEAN SAMPLE OF HIGHEST AMP. FOR EACH SPEAKER
	[~, mean_speaker_sample(x)] = max(summed_letter_speaker(:, x));
end

assert((all(fs == fs(1))), 'Error: not all sampling frequencies match') % check for equality fprintf(fs)
fs = fs(1); %change back into a singular value
end
