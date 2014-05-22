[ dropped_matrix, target_index, target_in_wheel ] = gen_drop_letterblock(letters_matrix, target_letter, target_per_block, oddball_drops )
%UNTITLED2 Summary of this function goes here
%   generate random dropped letters with criteria:
% 2 drops per wheel each loop
% each loop where target oddball is dropped will not have same numbered oddball in 
% other oddballs in same wheel will not be within 1 slot of the target oddball during target oddball loops
%oddballs_per_wheel = 2; 
[rep_block, cols] = size(letters_matrix);
dropped_row = ones(1, cols); %initialize the baseline 'on' row
target_index = find(strcmp(letters_matrix(1,:), target_letter));  %find the column number of the target letter
empty = isempty(target_index);  % boolean if target letter is not in the wheel 
target_in_wheel = ~empty;
dropped_matrix_components = []; %initialize matrix components

%%%%%%%
% creates possible rows of target oddball loops for two oddballs
% two drops can not be sequential with target
if target_in_wheel 
dropped_target_wheel = [];
for i = 1:cols
    dropped_row(target_index) = 0;  % create target dropped row
    if (i ~= target_index) 
        if (i ~= target_index + 1) 
            if (i ~= target_index - 1)
                dropped_row(i) = 0;
                dropped_target_wheel = [dropped_target_wheel; dropped_row];
            end
        end
    end
    dropped_row = ones(1, cols); %reset
end
[m, n] = size(dropped_target_wheel);
possible_indices = randperm(m, target_per_block);
%%%%


% two non-sequential oddballs
%neither drop can be the same number as the target oddball for that loop !!code later
for i = 1:cols  %loops through all letters (columns of the letters matrix)
    if target_in_wheel
    if (i ~= target_index)
        dropped_row(i) = 0;
%         for k = 1:cols
%             if j ~= i
%                 dropped_row(j) = 0; %second dropped per row
%             end
%         end
        for j = 1:(rep_block) % make block_size rows of each possible matrix
            dropped_matrix_components = [dropped_matrix_components; dropped_row];
        end
        dropped_row = ones(1, cols); %reset the dropped_row to 'on' in the oddball repeated loop 
    end
    end %change to else
    if empty
       dropped_row(i) = 0;
        for j = 1:(2 * rep_block) % make ten rows of each possible matrix
            dropped_matrix_components = [dropped_matrix_components; dropped_row];
        end
        dropped_row = ones(1, cols); %reset the dropped_row to 'on' in the oddball repeated loop 
    end
end
[m, n] = size(dropped_matrix_components);

if ~oddball_drops
    dropped_matrix_components = ones(m, n); %deletes non target oddballs if oddball_drops is 'off'
end

row_indices = randperm(m, rep_block); %of all possible non target components pick rep_block # of indices
dropped_matrix_init = zeros(rep_block, cols);  %initialize the dropped matrix components
for i= 1:length(row_indices)
    dropped_matrix_init(i, :) = dropped_matrix_components(row_indices(i), :); % fill in specified rows randomly created with non target components
end


% add in dropped_rows into the block at end
for i = 1:target_per_block
    dropped_matrix_init(rep_block - (i - 1), :) = dropped_target_wheel(possible_indices(i)); 
end

%%%%%
%reshuffle everything for final block matrix
row_indices = randperm(rep_block, rep_block); %get random ordering of the rows for reshuffling
for i = 1:rep_block
    dropped_matrix(i, :) = dropped_matrix_init(row_indices(i), :); % reshuffle all the rows of the matrix
end
%%%%

end

