function [ wheel_matrix, target_wheel_index, droppedLetter ] = assignLetters( possibleLetters, wheel_matrix_info, target_letter, target_cyc, tot_cyc, rearrangeCycles, ener_mask, subLetter)
% assignLetters takes all possible letters for  condition type, wheel_matrix_info which gives number of wheels and letters per wheel,  target_letter, number of target cycles per block,  total cycles in a block
% whether to rearrange letter ordering between trials in a block, ener_mask which either keeps all letters or replaces all targets with 'O's, and subLetter which is the

% RETURNS
% wheel_matrix where each cell represents a wheel, each rows represents a trial and each column represents the consecutive ordering of the letter within the trial
% target_wheel_index: wheel number of target wheel
% droppedLetter: only useful for tone_constant paradigm where pitch is assign according to the droppedLetter

% GENERAL PARAMETERS
letterO  = {'O'};
inner_cyc = tot_cyc - 2; % add in the first and end cycles last
wheel_num = length(wheel_matrix_info);

% CREATE GENERIC WHEELS, RECORD WHEEL AND LETTER INDEX OF  TARGET
index = 1;
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
sub_wheel = base_wheel_matrix{target_wheel_index};
droppedLetter = sub_wheel(1, target_letter_index);
if ener_mask
    for i = 1:wheel_num
        base_wheel_matrix{i} = repmat(letterO, 1, lettersWheel(i));
        sub_wheel_ener{i} = repmat(letterO, 1, lettersWheel(i));
    end
    base_wheel_matrix{target_wheel_index}(1, target_letter_index) = target_letter;
else
    sub_wheel(1, target_letter_index) = subLetter; % non_target wheel created
end

%CREATE RANDOMIZED ARRAY target_cyc_mat CORRESPONDING TO TARGET AND NON-TARGET CYCLES (1 0
%RESPECT.)
tryNext = 1;
while tryNext
    tryNext = 0;
    target_cyc_mat = reorder([ones(1, target_cyc) zeros(1, (inner_cyc - target_cyc))]);
    %FILTER OUT ALL SETS THAT CONTAIN CONSECUTIVE TARGET CYCLES
    for i = 1:(length(target_cyc_mat) - 1)
        if ((target_cyc_mat(i) == 1) && (target_cyc_mat(i + 1) == 1))
            tryNext = 1;
        end
    end
end

%ADD IN  NON TARGET CYCLES
target_cyc_mat = [0 target_cyc_mat 0];

% DEFINE WHEEL_MATRIX
wheel_matrix = cell(wheel_num, 1);
% for i = 1:wheel_num   
%     wheel_matrix{i} = zeros(tot_cyc, lettersWheel(i));
% end 

% PLACE WHEEL TYPES ACCORDING TO target_cyc_mat
for i = 1:wheel_num
    for j = 1:tot_cyc   
        % fprintf(strcat('new ', j))
        % j
        % i
        % [row,col] = size(wheel_matrix{i}) %++++++
        % [row,col] = size([(1:lettersWheel(i))]) %++++++
        % % [row,col] = size(base_wheel_matrix{i}) %++++++
        % [row,col] = size(sub_wheel_ener{i}) %++++++
        % % [row,col] = size(sub_wheel) %++++++
        if (i == target_wheel_index)
            if (target_cyc_mat(j) ==  1)
                wheel_matrix{i}(j,(1:lettersWheel(i))) = base_wheel_matrix{i};  % if its a target cycle add in a reorded set of generic wheel
            else
                if ener_mask
                    wheel_matrix{i}(j, (1:lettersWheel(i))) = sub_wheel_ener{i}; % reorder subbed out wheel for non-target cycles of target wheel  
                else
                    wheel_matrix{i}(j, (1:lettersWheel(i))) = sub_wheel; % reorder subbed out wheel for non-target cycles of target wheel
                end
            end
        else
            wheel_matrix{i}(j, (1:lettersWheel(i))) = base_wheel_matrix{i}; % reorder  generic template for non-target wheels
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
save total_target_letters  %for debugging
assert((target_cyc == total_target_letters), 'Error in generate_letter_matrix_wheel creating correct number of targets')
end


