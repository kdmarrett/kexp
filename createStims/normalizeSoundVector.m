function [output] = normalizeSoundVector(input)
	% take a vector and normalize between -1 and 1 to prevent clipping for WAV files

[rows, ~] = size(input); % input is your matrix
% colMax = max(abs(input), [], 1); % take max absolute value to account for negative numbers
% output = input ./ repmat(colMax, rows, 1);
minval = min(input);
input = input - repmat(minval, rows, 1);
maxval = max(input);
input = 2 * (input ./ repmat(maxval, rows, 1));
output = .99 * (input - repmat(1, rows, 1));
maxval = max(output);
minval = min(output);
assert(((abs(maxval) <= 1) & (abs(minval) <= 1)), 'Error: out of range')
end
