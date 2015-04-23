function [output] = reorderDisplaced (input)
%reorders but keeps displaced letter placement for each row

[rows, cols] = size(input);

% keep the first row the same to make visual primer
output(1,:) = input(1,:);

for i = 2:rows
	inputRow = input(i, :);
	%ind = randi([2, cols]);
	%firstHalf = inputRow(ind:cols);
	%secondHalf = inputRow(1:(ind - 1));
    %newRow = [firstHalf, secondHalf];
    newRow = inputRow(randperm(length(inputRow)));
    assert(length(newRow) == cols)
	output(i, :) = newRow;
end

