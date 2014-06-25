
function [ fs, final_letter_path ] = trimLetters(letter_samples, letter_path, letterArray, pitches )
% Writes all new letters with the specified sample length of letter_samples and saves into 
% folder final_letter_path a subdirectory of letter_path
% fs = 24414 letter sample rate

letterSound = {};
trimmedLetters = {};
for i= 1:length(pitches.all) %loop through all semitone folders
    for j = 1:length(letterArray) %loop through all letters in each semitone folder
        [letterSound{j}, fs] = wavread(fullfile(letter_path, pitches.all{i}, letterArray.alphabetic{j}));  % create a cell array holding all the letter wavs for the specific i^th pitches.all
    end
    
    % EXCEPTIONS 'C', 'W'
    c = letterSound{3}; % makes c more audible
    [row, col] = size(c);
    letterSound{3}=c(1500:row);
    w = letterSound{23};
    letterSound{23} = w(1:4000);

    for j = 1:length(letterArray)
        letterVector = letterSound{j};
        [rows, cols] = size(letterVector);
        if rows < letter_samples
            temp = zeros(letter_samples, 1);
            for k = 1:rows
                temp(k) = letterVector(k);
            end
            trimmedLetters{j} = temp;
        else
            trimmedLetters{j} = letterVector(1:letter_samples, :);
        end
        wavwrite(trimmedLetters{j}, fs, fullfile(letter_path, int2str(letter_samples), pitches.all{i}, letterArray{j}));
    end
end
final_letter_path= fullfile(letter_path, int2str(letter_samples));
end

