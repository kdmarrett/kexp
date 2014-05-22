function [ wheel_matrix, target_wheel_index, wheel_col, total_target_letters ] = generate_letter_matrix_wheel( letterArray, wheel_num, letters_wheel, target_letter, targ_cyc, tot_cyc)

%wheel matrix is a cell array where each array contains rows which
%represent the trials or blocks for each wheel, the columns represent the
%order in which the letter will be played in which is why they are randomly
%reordered at the end.

subLetter = {'Z'};
inner_cyc = tot_cyc - 2; % add in the first and end cycles last

%create the generic wheels and record the wheel and letter number of
%the target
for i=1:wheel_num
    base_wheel_matrix{i} = letterArray((1 + (letters_wheel * (i - 1))):(letters_wheel * i)); 
if sum(sum(strcmp(base_wheel_matrix{i}, target_letter))) %if this wheel contains the target letter
    target_letter_index = find(strcmp(base_wheel_matrix{i}, target_letter), 1); %record the letter index
    target_wheel_index = i;  %record the wheel index
end %else?
end

%create the target wheel without the target letter
sub_wheel = base_wheel_matrix{target_wheel_index};
sub_wheel(1, target_letter_index) = subLetter;

%create an array corresponding to target and non-target cycles 
%filter out all sets that contain the target letter
tryNext = 1;
while tryNext
    tryNext = 0;
    targ_cyc_mat = reorder([ones(1, targ_cyc) zeros(1, (inner_cyc - targ_cyc))]);
    for i = 1:(length(targ_cyc_mat) - 1)  %tests for consecutive target cycles
        if ((targ_cyc_mat(i) == 1) && (targ_cyc_mat(i + 1) == 1))
            tryNext = 1;
        end
    end
end

%add in the non target cycles
targ_cyc_mat = [0 targ_cyc_mat 0];

for i = 1:wheel_num
    for j = 1:length(targ_cyc_mat)
        if (i == target_wheel_index)
            if (targ_cyc_mat(j) ==  1)
                wheel_matrix{i}(j, :) = reorder(base_wheel_matrix{i});  % if its a target cycle add in a reorded set of the generic wheel
            else
                wheel_matrix{i}(j, :) = reorder(sub_wheel); % reorder the subbed out wheel for non-target cycles of the target wheel
            end
        else
            wheel_matrix{i}(j, :) = reorder(base_wheel_matrix{i}); % reorder the generic template for non-target wheels
        end
    end
end
 wheel_matrix{target_wheel_index};
 temp = strcmp(target_letter, wheel_matrix{target_wheel_index});
 total_target_letters = sum(sum(temp));
 [wheel_row, wheel_col] = size(wheel_matrix{1});

