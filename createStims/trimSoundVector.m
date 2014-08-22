function [output] = trimSoundVector(input, fs, new_length, gate_start, gate_end)
% trims a vector by the length specified by new_length (starting from the beginning in samples) and gates the trimmed ends to prevent popping

[rows, cols] = size(input);
if rows < new_length
    temp = zeros(new_length, 1); % add zeros to the end of the letter
    for k = 1:rows
        temp(k) = input(k);
    end
    output = temp;
else
    output = input(1:new_length, :); % truncate vector at row new_length
end
output = createGate(output, fs, gate_start, gate_end);
[rows, cols] = size(output);
assert((rows == new_length), 'Error in trimLetters: not all letters equal to new_length')
end