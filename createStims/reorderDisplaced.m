function [output] = reorderDisplaced (input, targ_cyc_ind, ...
targ_wheel_bool, target_letter_index)
%reorders but ensure it's not the last letter of any cycle

[rows, cols] = size(input);

% keep the first row the same to make visual primer
output(1,:) = input(1,:);

for i = 2:rows
	inputRow = input(i, :);
    newOrder = randperm(length(inputRow));
    % this prevents two targets in a row
    % this also guarantees pupillometry is recording
    % for more time after a target
    if (targ_wheel_bool && any(i == targ_cyc_ind))
        while (newOrder(length(newOrder)) ~= ...
            target_letter_index)
            newOrder = randperm(length(inputRow));
        end
    end
    newRow = inputRow(newOrder);
    assert(length(newRow) == cols)
	output(i, :) = newRow;
end

