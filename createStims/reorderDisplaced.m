function [output] = reorderDisplaced (input)
%reorders but keeps displaced letter placement for each row

[rows, cols] = size(input);

for i = 1:rows
	inputRow = input(i, :);
	ind = randi([2 (cols - 1)]);
	firstHalf = inputRow(ind:cols);
	secondHalf = inputRow(1:(ind - 1));
	output(i, :) = [firstHalf secondHalf];
end

