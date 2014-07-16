function [] = trimInstrNotes( fs, instrNote_dir, letter_samples, pitches, instrument_dynamics, env_instrNotes, instr_list, speaker_list, letterEnvelope, list_of_pitches, force_recreate, letterArray, envelope_type, mean_speaker_sample, start_sample_one)

if ~env_instrNotes
	note_length = ceil(letter_samples / 3);
	half_note = ceil(note_length / 2);
	x = linspace(0, pi, note_length);
	y = ((sin(x)).^2)';
	% plot(x, y)
	% waitforbuttonpress
	% close
	envelope_base = zeros(letter_samples, 1);
	envelope = zeros(letter_samples, 3);
	for i = 1:length(speaker_list)
		if ((mean_speaker_sample(i) > half_note ) & ~start_sample_one)
			start_sample = ceil(mean_speaker_sample(i)) - half_note % keep peak of note the mean for each speaker
			chi = [zeros((start_sample - 1), 1); y; zeros((letter_samples - (start_sample + note_length + 1)), 1)];
			[mstart, nstart] = size(chi);
			envelope(:, i) = [zeros((start_sample - 1), 1); y; zeros((letter_samples - (start_sample + note_length + 1)), 1)]
		else
			start_sample = 1;
			chi = [y; zeros((letter_samples - (note_length)), 1)];
			[m1, n1] = size(chi);
			envelope(:, i) = [y; zeros((letter_samples - (note_length)), 1)];
		end
		% size(envelope(:, i))
		% sum(envelope(:, i))
		% envelope(:, i) = [zeros(1:)]
		% [m,n] = size(y)
		% size(y)
		% size(envelope_base((start_sample:(length(y))), 1))
		% envelope_base((start_sample:(length(y) - 1)), 1) = y;
        % envelope(:, i) = envelope_base;

        % VISUALIZE:
        % plot(envelope(:, i))
        % waitforbuttonpress
	end
end
for i = 1:length(speaker_list)
	for j = 1:length(instr_list)
		for k = 1:length(list_of_pitches)
			for l = 1:length(letterArray.alphabetic)
				output_path = fullfile(instrNote_dir, 'trim', speaker_list{i}, instr_list{j}, list_of_pitches{k}, envelope_type, letterArray.alphabetic{l});
			    if ((~exist(output_path, 'dir')) || force_recreate)
			    	if env_instrNotes
						final_envelope = letterEnvelope{l};
						assert(strcmpi(envelope_type, 'env'), 'Error: envelope type and env_instrNotes do not match')
					else
						final_envelope = envelope(:, i);
					end
			    	createStruct(output_path);
			    	note = semitoneToNote(list_of_pitches{k}, pitches);
			    	fn = strcat(instr_list{j}, '.', instrument_dynamics, '.', note, '.wav');
			    	% fullfile(instrNote_dir, 'raw', instr_list{j}, fn) 
			    	[instrNote, fs, nbits] = wavread(fullfile(instrNote_dir, 'raw', instr_list{j}, fn), letter_samples);
			    	        
	    	        % VISUALIZE:
			        % plot(final_envelope)
			        % title(letterArray.alphabetic{l})
			        % waitforbuttonpress
			        % close

			    	final_instrNote = final_envelope .* instrNote;
			    	if env_instrNotes
				    	trimInstrNotes_path = fullfile(instrNote_dir, 'trim', speaker_list{i}, instr_list{j}, list_of_pitches{k}, envelope_type, letterArray.alphabetic{l}, fn); % +++
				    else
				    	trimInstrNotes_path = fullfile(instrNote_dir, 'trim', speaker_list{i}, instr_list{j}, list_of_pitches{k}, envelope_type, fn); % pitch and instrument is redundant
				    end
			    	final_instrNote = normalizeSoundVector(final_instrNote);
			    	final_instrNote = trimSoundVector(final_instrNote, fs, letter_samples, 1, 1);
			    	% plot(final_instrNote)
			    	% sound(final_instrNote, 16000)
			    	% waitforbuttonpress
			    	wavwrite( final_instrNote, fs, nbits, trimInstrNotes_path )
			    	assert(length(final_instrNote) == letter_samples, 'Error: instrNote does not match letter length')
			    end
			end
		end
	end
end
end



