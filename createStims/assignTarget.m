function [replacement_letter, target_letter, target_wheel, location_code, block_no, ...
blocktrial, base_wheel_matrix] = assignTarget(trial_no, ...
possible_letters, wheel_matrix_info, blocktrial, left_ind, mid_ind, ...  
right_ind, cond_no);

if (trial_no <= 1)
	block_no = 1;
	blocktrial(1) = blocktrial(1) + 1;
elseif (trial_no <= 10)
	block_no = 2;
	blocktrial(2) = blocktrial(2) + 1;
elseif (trial_no <= 19)
	block_no = 3;
	blocktrial(3) = blocktrial(3) + 1;
elseif (trial_no <= 28)
	block_no = 4;
	blocktrial(4) = blocktrial(4) + 1;
elseif (trial_no == 29) 
	block_no = 5;
	blocktrial(5) = blocktrial(5) + 1;
else
	block_no = 5 + cond_no;
	blocktrial(block_no) = blocktrial(block_no) + 1;
end

index = 1;
wheel_num = length(wheel_matrix_info);

% distribute letters
for i=1:wheel_num
    lettersWheel(i) = wheel_matrix_info(i);
	base_wheel_matrix{i} = possible_letters(index:((index +...
		lettersWheel(i) - 1))); 
	index = index + lettersWheel(i);
end

target_wheel = [];
if (any(trial_no == left_ind))
	location_code = 'l';
	target_wheel = 1;
elseif (any(trial_no == mid_ind))
	location_code = 'm';
	target_wheel = 2;
else 
	location_code = 'r';
	target_wheel = 3;
end

%choose replacement letter
temp = randi(wheel_matrix_info(target_wheel));
target_letter = base_wheel_matrix{target_wheel}(1, temp);
rep_ind = randi(wheel_matrix_info(target_wheel));
while (rep_ind == temp)
    rep_ind = randi(wheel_matrix_info(target_wheel));
end
replacement_letter = base_wheel_matrix{target_wheel}(1, rep_ind);

end
