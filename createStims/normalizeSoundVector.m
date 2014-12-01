function [output] = normalizeSoundVector(input)
	% take a vector and normalize between -1 and 1 to prevent clipping for WAV files

[rows, ~] = size(input); % input is your matrix
maxval = max(abs(input));
output = .99 * (input ./ repmat(maxval, rows, 1));
maxval = max(output);
minval = min(output);
%assert(((abs(maxval) <= 1) && (abs(minval) <= 1)), 'Error: out of range')
end
