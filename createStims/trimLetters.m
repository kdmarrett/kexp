function [ fs, final_letter_path ] = trimLetters(letter_samples, letter_path, letterArray, pitches, recreate_trimmed_letters, speakers )
% Writes all new letters with the specified sample length of letter_samples and saves into 
% folder final_letter_path a subdirectory of letter_path
    
for x = 1:length(speakers)
    fs = 24414; % default letter sample rate
    shifted_letter_path = fullfile(letter_path, 'shiftedLetters', speakers{x});
    final_letter_path = fullfile(letter_path, finalShiftTrimLetters, int2str(letter_samples));
    output_path = fullfile(final_letter_path, speakers{x});
    if ((~exist(final_letter_path, 'dir')) || recreate_trimmed_letters)
        letterSound = {};
        trimmedLetters = {};
        for i= 1:length(pitches.all) %loop through all possible semitones
            for j = 1:length(letterArray.alphabetic) %loop through all letters in each semitone dir
                [letterSound{j}, fs] = wavread(fullfile(shifted_letter_path, pitches.all{i}, letterArray.alphabetic{j}));  % letter wavs for each semitone
            end
            
            % EXCEPTIONS 'C', 'W'
            c = letterSound{3}; % makes c more audible
            [row, col] = size(c);
            letterSound{3}=c(1500:row);
            w = letterSound{23};
            letterSound{23} = w(1:4000); % 'W' to "dub"

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
                trimmedLetters{j} = createGate(trimmedLetters{j}, fs, 0, 1);
                [rows, cols] = size(trimmedLetters{j});
                assert((rows == letter_samples), 'Error in trimLetters: not all letters equal to letter_samples')
                createStruct(fullfile(output_path, pitches.all{i}))
                wavwrite(trimmedLetters{j}, fs, fullfile(output_path, pitches.all{i}, strcat(letterArray.alphabetic{j}, '.wav')));
            end
        end
    end
end
end



