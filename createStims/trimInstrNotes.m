function [] = trimInstrNotes( fs, instrNote_dir, letter_samples, pitches, instrument_dynamics, env_instrNotes, instr_list, speaker_list, letterEnvelope, list_of_pitches, force_recreate, letterArray, envelope_type, mean_speaker_sample, start_sample_one, start_semitone_index, wheel_matrix_info)

start_point = 4500;
	if env_instrNotes
		for i = 1:length(speaker_list)
			for j = 1:length(instr_list)
				for k = start_semitone_index(j):(start_semitone_index(j) + wheel_matrix_info(j) - 1)  % check
					for l = 1:length(letterArray.alphabetic)
						output_path = fullfile(instrNote_dir, 'trim', envelope_type, speaker_list{i}, instr_list{j}, list_of_pitches{k}, letterArray.alphabetic{l});
						if ((~exist(output_path, 'dir')) || force_recreate)
							final_envelope = letterEnvelope{l};
							assert(strcmpi(envelope_type, 'env'), 'Error: envelope type and env_instrNotes do not match')
						    	createStruct(output_path);
						    	note = semitoneToNote(list_of_pitches{k}, pitches);
						    	fn = strcat(instr_list{j}, '.', instrument_dynamics, '.', note, '.wav'); 
					    	        	if (j == 1)   % piano has late start
						    		[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), [start_point (start_point + letter_samples - 1) ] );
						    	end
						   	if (j == 3) % piano has early start
							    	[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), letter_samples );
							    end
							if (j == 2)	
							    	[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), [start_point (start_point + letter_samples - 1) ] );
							end
						    	final_instrNote = final_envelope .* instrNote;
						    	trimInstrNotes_path = fullfile(instrNote_dir, 'trim', envelope_type, speaker_list{i}, instr_list{j}, list_of_pitches{k}, letterArray.alphabetic{l}, fn); 
						    	final_instrNote = normalizeSoundVector(final_instrNote);
						    	final_instrNote = trimSoundVector(final_instrNote, fs, letter_samples, 1, 1);
						    	wavwrite( final_instrNote, fs, nbits, trimInstrNotes_path )
						    	assert(length(final_instrNote) == letter_samples, 'Error: instrNote does not match letter length')
						end
					end
				end
			end
		end
	end
	if ~env_instrNotes

		% CREATE SIN ENVELOPE
		note_length = ceil(letter_samples / 3);
		half_note = ceil(note_length / 2);
		x = linspace(0, pi, note_length);
		y = ((sin(x)).^3)';
		envelope_base = zeros(letter_samples, 1);
		envelope = zeros(letter_samples, 3);
		for i = 1:length(speaker_list)
			if ((mean_speaker_sample(i) > half_note ) & ~start_sample_one)
				start_sample = ceil(mean_speaker_sample(i)) - half_note % keep peak of note the mean for each speaker
				if (i == 3) %start marimba at 1 always to capture some of the initial articulation
					start_sample = 1;
				end
				envelope(:, i) = [zeros((start_sample - 1), 1); y; zeros((letter_samples - (start_sample + note_length + 1)), 1)]
			else
				start_sample = 1;
				envelope(:, i) = [y; zeros((letter_samples - (note_length)), 1)];
			end
		end

		for j = 1:length(instr_list)
			for k = (start_semitone_index(j):(start_semitone_index(j) + wheel_matrix_info(j) - 1) ) 
				for l = 1:length(letterArray.alphabetic)
					output_path = fullfile(instrNote_dir, 'trim', envelope_type, instr_list{j}, list_of_pitches{k}, letterArray.alphabetic{l});
					if ((~exist(output_path, 'dir')) || force_recreate)
						final_envelope = envelope(:, i);
					    	createStruct(output_path);
					    	note = semitoneToNote(list_of_pitches{k}, pitches);
					    	fn = strcat(instr_list{j}, '.', instrument_dynamics, '.', note, '.wav');
				    	        	if (j == 1)   % piano has late start
					    		[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), [start_point (start_point + letter_samples - 1) ] );
					    	end
					   	if (j == 3) % piano has early start
						    	[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), letter_samples );
						end
						if (j == 2)
						    	[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), [start_point (start_point + letter_samples - 1) ] );
						end
					    	fn = strcat( note, '.wav');
					    	final_instrNote = final_envelope .* instrNote;
					    	trimInstrNotes_path = fullfile(instrNote_dir, 'trim', envelope_type, instr_list{j}, list_of_pitches{k}, fn); % pitch and instrument is redundant
					    	final_instrNote = normalizeSoundVector(final_instrNote);
					    	final_instrNote = trimSoundVector(final_instrNote, fs, letter_samples, 1, 1);
					    	wavwrite( final_instrNote, fs, nbits, trimInstrNotes_path )
					    	assert(length(final_instrNote) == letter_samples, 'Error: instrNote does not match letter length')
					end
				end
			end
		end
	end
end



