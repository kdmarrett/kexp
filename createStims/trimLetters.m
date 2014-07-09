function [ fs, trim_letter_path ] = trimLetters(letter_samples, letter_path, letterArray, pitches, recreate_trimmed_letters, speaker_list, version_num, speaker_amp_weights )
% Writes all new letters with the specified sample length of letter_samples and saves into 
% folder trim_letter_path a subdirectory of letter_path
fs = zeros(3,1);    
for x = 1:length(speaker_list)
    % fs = 24414; % default letter sample rate
    shifted_letter_path = fullfile(letter_path, 'shiftedLetters', speaker_list{x});
    trim_letter_path = fullfile(letter_path, 'finalShiftTrimLetters', int2str(letter_samples));
    output_path = fullfile(trim_letter_path, speaker_list{x});
    if ((~exist(trim_letter_path, 'dir')) || recreate_trimmed_letters)
        letterSound = {};
        trimmedLetters = {};
        for i= 1:length(pitches.all) %loop through all possible semitones
            for j = 1:length(letterArray.alphabetic) %loop through all letters in each semitone dir
                fp = fullfile(shifted_letter_path, pitches.all{i});
                temp_fn = strcat(speaker_list{x}, '-', letterArray.alphabetic{j}, int2str(version_num), '-t', '.wav');
                if (strcmpi(speaker_list{x}, 'Original') || strcmpi(speaker_list{x}, 'male_trimmed') || strcmpi(speaker_list{x}, 'female_trimmed'))
                    fn = letterArray.alphabetic{j};
                elseif exist(fullfile(fp, temp_fn), 'file')
                    fn = temp_fn;
                else
                    fn = strcat(speaker_list{x}, '_', letterArray.alphabetic{j}, int2str(version_num), '.wav');
                end
                ff = fullfile(fp, fn); 
                [letterSound{j}, fs_speaker] = wavread(ff);  % letter wavs for each semitone
                letterEnvelope{j} = envelope(letterSound{j}); 
                plot(letterSound{j})
                hold on
                plot(letterEnvelope{j}, 'r')
                title(letterArray.alphabetic{j})
                waitforbuttonpress
                hold off

                % CHANGE OVERALL AMPLITUDE OF INDIVIDUAL SPEAKERS
                letterSound{j} = speaker_amp_weights(x) .* letterSound{j};
            end

            % % EXCEPTIONS 'C', 'W'
            % c = letterSound{3}; % makes c more audible
            % [row, col] = size(c);
            % letterSound{3}=c(1500:row);
            % w = letterSound{23};
            % % letterSound{23} = w(1:4200); % 'W' to "dub"

            % TRIM EACH LETTER 
            for j = 1:length(letterArray.alphabetic)
                letterVector = letterSound{j};
                [rows, cols] = size(letterVector);
                if rows < letter_samples
                    temp = zeros(letter_samples, 1); % add zeros to the end of the letter
                    for k = 1:rows
                        temp(k) = letterVector(k);
                    end
                    trimmedLetters{j} = temp;
                else
                    trimmedLetters{j} = letterVector(1:letter_samples, :); % truncate vector at row letter_samples
                end
                trimmedLetters{j} = createGate(trimmedLetters{j}, fs_speaker, 0, 1);
                [rows, cols] = size(trimmedLetters{j});
                assert((rows == letter_samples), 'Error in trimLetters: not all letters equal to letter_samples')
                createStruct(fullfile(output_path, pitches.all{i}))
                wavwrite(trimmedLetters{j}, fs_speaker, fullfile(output_path, pitches.all{i}, strcat(letterArray.alphabetic{j}, '.wav')));
            end
        end
    end
    % assert((fs_speaker == 16000), 'Error: incorrect wavread of letterSound')
    fs(x) = fs_speaker;
end
assert((all(fs == fs(1))), 'Error: not all sampling frequencies match') % check for equality fprintf(fs)
fs = fs(1); %change back into a singular value
end



