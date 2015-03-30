[target_letter, location_code, block_no, blocktrial] = assignTarget(trial_no, ...
possible_letters, wheel_matrix_info, blocktrial);

left_ind  = [1 4:13 34]; 
mid_ind   = [2 14:23 35];
right_ind = [3 24:33 36];

if (trial_no =< 3)
	block_no = 1;
	blocktrial(1) = blocktrial(1) + 1;
elseif (trial_no =< 13)
	block_no = 2;
	blocktrial(2) = blocktrial(2) + 1;
elseif (trial_no =< 23)
	block_no = 3;
	blocktrial(3) = blocktrial(3) + 1;
elseif (trial_no =< 33)
	block_no = 4;
	blocktrial(4) = blocktrial(4) + 1;
else 
	block_no = 5;
	blocktrial(5) = blocktrial(5) + 1;
end

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

if (any(trial_no == left_ind))
	temp = randi(wheel_matrix_info(1));
	target_letter = base_wheel_matrix{1}(1, temp);
	location_code = 'l';
elseif (any(trial_no == mid_ind))
	temp = randi(wheel_matrix_info(2));
	target_letter = base_wheel_matrix{2}(1, temp);
	location_code = 'm';
else 
	temp = randi(wheel_matrix_info(3));
	target_letter = base_wheel_matrix{3}(1, temp);
	location_code = 'r';
end

