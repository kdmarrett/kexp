function [ wheel_matrix, targ_cyc_ind, target_letter_index ] = assignLetters( replacement, ...
	possibleLetters, wheel_matrix_info, target_letter,... 
	target_cycles, tot_cyc, rearrangeCycles, ener_mask, ...
	base_wheel_matrix, target_wheel_index)

% assignLetters takes all possible letters for  condition type,
% wheel_matrix_info which gives number of wheels and letters per wheel,
% target_letter, number of target cycles per trial,  total cycles in a
% block whether to rearrange letter ordering between trials in a block,
% ener_mask which either keeps all letters or replaces all targets with
% 'O's

% RETURNS
% wheel_matrix where each cell represents a wheel, each rows represents
% a trial and each column represents the consecutive ordering of the
% letter within the trial target_wheel_index: wheel number of target
% wheel droppedLetter: only useful for tone_constant paradigm where
% pitch is assign according to the droppedLetter

% GENERAL PARAMETERS
wheel_num = length(wheel_matrix_info);

% CREATE GENERIC WHEELS, RECORD WHEEL AND LETTER INDEX OF TARGET
index = 1;
target_letter_index = find(strcmp(base_wheel_matrix{target_wheel_index}, target_letter), 1); %record  letter index

% CREATE TARGET WHEEL WITHOUT TARGET LETTER
replacement_wheel = base_wheel_matrix{target_wheel_index};
replacement_wheel(1, target_letter_index) = replacement;

% CREATE TARGET WHEEL WITH O's
% if ener_mask
%     for i = 1:wheel_num
%         base_wheel_matrix{i} = repmat(letterO, 1, lettersWheel(i));
%     end
%     base_wheel_matrix{target_wheel_index}(1, target_letter_index) = target_letter;
% end

% DECLARE WHEEL_MATRIX
wheel_matrix = cell(wheel_num, 1);

targ_cyc_ind = [];
while(length(targ_cyc_ind) ~= target_cycles)
	% not the first cycle
	temp = randi(4) + 1;
	if (any(targ_cyc_ind == temp))
		continue;
	end
	targ_cyc_ind = [ targ_cyc_ind temp]; 
end
assert(length(targ_cyc_ind) == target_cycles)

% PLACE WHEEL TYPES
for i = 1:wheel_num
    for j = 1:tot_cyc   
		% reorder generic template for non-target wheels
		if (i == target_wheel_index)
			if any(targ_cyc_ind == j)
				wheel_matrix{i}(j, (1:wheel_matrix_info(i))) = base_wheel_matrix{i}; 
			else
				wheel_matrix{i}(j, (1:wheel_matrix_info(i))) = replacement_wheel; 
			end
		else
			wheel_matrix{i}(j, (1:wheel_matrix_info(i))) = base_wheel_matrix{i}; 
		end
    end
end

% CHANGE ORDERING BETWEEN CYCLES
if rearrangeCycles
    for i = 1:length(wheel_matrix)
        targ_wheel_bool = (i == target_wheel_index);
        [wheel_matrix{i}] = reorderDisplaced(wheel_matrix{i}, ...
            targ_cyc_ind, targ_wheel_bool, target_letter_index);
    end
end

% TEST FOR CORRECT TARGETS
match_matrix = strcmp(target_letter, wheel_matrix{target_wheel_index});
total_target_letters = sum(sum(match_matrix));
assert((target_cycles == total_target_letters), ...
'Error in generate_letter_matrix_wheel creating correct number of targets')
end


