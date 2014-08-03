function [ letter_to_pitch ] = assignConstantPitch( possibleLetters, total_letters, total_pitches, wheel_matrix_info, pitch_wheel )
%Puts all possible letters into a cell where the column decides which pitch
%is assigned to the letter

letter_index = 1;
for i = 1:length(wheel_matrix_info)

	wheel_letters{i} = cell(1, wheel_matrix_info(i));
	wheel_index = 1;
	for j =(letter_index):((letter_index - 1) + wheel_matrix_info(i));
		wheel_letters{i}{wheel_index} = possibleLetters{j};
		wheel_index = wheel_index + 1;
	end
	letter_index = letter_index + wheel_matrix_info(i);

	%CREATE A VECTOR OF SHUFFLED INDICES
	mixed_indices = randperm(wheel_matrix_info(i), wheel_matrix_info(i));

	%CREATE CELLS
	letter_to_pitch{i} = cell(1, wheel_matrix_info(i));
	for j = 1:length(mixed_indices)
		letter_to_pitch{i}{1, j} = wheel_letters{i}{mixed_indices(j)};
	end
end
end

