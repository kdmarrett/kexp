function [ wheel_matrix, target_wheel_index ] = assignLetters( ...
	possibleLetters, wheel_matrix_info, target_letter,... 
	target_cycles, tot_cyc, rearrangeCycles, ener_mask)

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
letterO  = {'O'}; 
wheel_num = length(wheel_matrix_info);

if strcmp(letterO, target_letter)
	replacement = {'R'};
else
	replacement = {'O'};
end

% CREATE GENERIC WHEELS, RECORD WHEEL AND LETTER INDEX OF TARGET
index = 1;
target_wheel_index = [];
for i=1:wheel_num
    lettersWheel(i) = wheel_matrix_info(i);
    base_wheel_matrix{i} = possibleLetters(index:((index + lettersWheel(i) - 1)));
    if sum(sum(strcmp(base_wheel_matrix{i}, target_letter))) %if this wheel contains  target letter
        target_letter_index = find(strcmp(base_wheel_matrix{i}, target_letter), 1); %record  letter index
        target_wheel_index = i;  %record  wheel index
    end 
    index = index + lettersWheel(i);
end
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
for i = 1:target_cycles
	% not the first cycle
	temp = randi(4) + 1;
	targ_cyc_ind = [ targ_cyc_ind temp]; 
end

% PLACE WHEEL TYPES
for i = 1:wheel_num
    for j = 1:tot_cyc   
		% reorder generic template for non-target wheels
		if (any(targ_cyc_ind == j) && (i == target_wheel_index))
			wheel_matrix{i}(j, (1:lettersWheel(i))) = replacement_wheel; 
			target_wheel_index
		else
			wheel_matrix{i}(j, (1:lettersWheel(i))) = base_wheel_matrix{i}; 
		end
    end
end

% CHANGE ORDERING BETWEEN CYCLES
if rearrangeCycles
    for i = 1:length(wheel_matrix)
        [wheel_matrix{i}] = reorderDisplaced(wheel_matrix{i});
    end
end

% TEST FOR CORRECT TARGETS
match_matrix = strcmp(target_letter, wheel_matrix{target_wheel_index});
total_target_letters = sum(sum(match_matrix));
target_cycles
total_target_letters
assert((target_cycles == total_target_letters), ...
'Error in generate_letter_matrix_wheel creating correct number of targets')
end


