function [ letters_matrix ] = generateLettersMatrixKarl(trial_number, block_size)
%UNTITLED4 Summary of this function goes here
%   trial number
 letterArray = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
 letters_Matrix = cell(block_size, 5 * trial_number);

for i = 1:block_size
    letters_matrix(i, :)= letterArray(1:5*trial_number);
end

