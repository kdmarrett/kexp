function [output] = reorderDisplaced (input)
%reorders but keeps displaced letter placement for each row

[rows, cols] = size(input);

for i = 2:rows
	inputRow = input(i, :);
	ind = randi([2, (cols)]);
	firstHalf = inputRow(ind:cols);
	secondHalf = inputRow(1:(ind - 1));
    newRow = [firstHalf, secondHalf];
    assert(length(newRow) == length(cols))
	output(i, :) = newRow;
end

