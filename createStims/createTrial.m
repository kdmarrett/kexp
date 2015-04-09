function [] = createTrial(tot_trial, white_noise_decibel, preblock,
    wheel_matrix_info, speaker_list, trim_letter_path, tot_wheel,
    wheel_matrix, )

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
                [L, R] = stimuliHRTF(combined_sound, fs, angle, ...
                    distance_sound, K70_dir);
                combined_sound_proc = (10 ^(letter_decibel / 20)) * [L R];
                
                % % RECORD TARGET TIME
                if strcmp(letter, target_letter)
                    target_sample_index = wheel_sample_index + ...
                     track_sample_index;
                    target_time = [target_time (target_sample_index / fs)];
                    if tone_constant %check that target tone is constant
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
