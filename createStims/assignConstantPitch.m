function [ letter_to_pitch ] = assignConstantPitch( letterArray, total_letters, total_pitches, subLetter, droppedLetter )
%Puts all possible letters into a cell where the column decides which pitch
%is assigned to the letter

%GET/ASSIGN ROUGH ESTIMATES OF CELL SIZE
ratio = floor(total_letters / total_pitches);
left_over = (total_letters - (ratio * total_pitches));
extra_rows = 3; 
ltp_rows = ratio + extra_rows;

%CREATE A VECTOR OF SHUFFLED INDICES
mixed_indices = randperm(total_letters, total_letters);

%CREATE CELLS
letter_to_pitch = cell(ltp_rows, total_pitches);

col_indexer = 1;
row_indexer = 1;
for i = 1:length(mixed_indices)
    letter_to_pitch{row_indexer, col_indexer} = letterArray{mixed_indices(i)};
    col_indexer = col_indexer + 1;
    if (col_indexer > total_pitches)
        col_indexer = 1;
        row_indexer = row_indexer + 1;
    end
end

% PLACE SUBLETTER INTO LETTER_TO_PITCH ARRAY
columnSubLetter = find(sum(strcmp(droppedLetter, letter_to_pitch))); %find the column of the dropped out letter
subRow = (ratio + extra_rows);
columnSubLetter;
letter_to_pitch{subRow, columnSubLetter} = subLetter{1}; %add in the subbed letter to that column
end

