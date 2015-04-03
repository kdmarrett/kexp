function [target_letter, target_wheel, location_code, block_no, ...
blocktrial, base_wheel_matrix] = assignTarget(trial_no, ...
possible_letters, wheel_matrix_info, blocktrial, left_ind, mid_ind, ...  
right_ind);

if (trial_no <= 3)
	block_no = 1;
	blocktrial(1) = blocktrial(1) + 1;
elseif (trial_no <= 12)
	block_no = 2;
	blocktrial(2) = blocktrial(2) + 1;
elseif (trial_no <= 21)
	block_no = 3;
	blocktrial(3) = blocktrial(3) + 1;
elseif (trial_no <= 30)
	block_no = 4;
	blocktrial(4) = blocktrial(4) + 1;
else 
	block_no = 5;
	blocktrial(5) = blocktrial(5) + 1;
end

index = 1;
target_wheel_index = [];
wheel_num = length(wheel_matrix_info);

% distribute letters
for i=1:wheel_num
    lettersWheel(i) = wheel_matrix_info(i);
    base_wheel_matrix{i} = possible_letters(index:((index + lettersWheel(i) - 1)));
    index = index + lettersWheel(i);
end

target_wheel = [];
if (any(trial_no == left_ind))
	temp = randi(wheel_matrix_info(1));
	target_letter = base_wheel_matrix{1}(1, temp);
	location_code = 'l';
	target_wheel = 1;
elseif (any(trial_no == mid_ind))
	temp = randi(wheel_matrix_info(2));
	target_letter = base_wheel_matrix{2}(1, temp);
	location_code = 'm';
	target_wheel = 2;
else 
	temp = randi(wheel_matrix_info(3));
	target_letter = base_wheel_matrix{3}(1, temp);
	location_code = 'r';
	target_wheel = 3;
end

