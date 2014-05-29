function [ output ] = reorder( input )
%reorder randomly shuffles the elements of any matrix or array

[rows,cols] = size(input);
if iscell(input)
    output = cell(rows, cols);
else
    output = zeros(rows, cols);
end
for i = 1:rows
    ind = randperm(cols);
    for j = 1:cols
        if iscell(input)
            output{i, j} = input(i, ind(j));
        else
            output(i, j) = input(i, ind(j));
        end
    end
end
end

