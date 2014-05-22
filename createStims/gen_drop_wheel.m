function [ dropped_matrix, target_index, target_wheel_index ] = gen_drop_wheel( wheel_matrix, target_letter, target_per_block, oddball_drops)
%UNTITLED2 Summary of this function goes here
%   generate random dropped letters


 letterArray = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
wheel_matrix
%breaks for multiple letters
for i = 1: length(wheel_matrix)
    for j = 1: length(target_letter)
        wheel_matrix{i}
        [dropped_matrix{i}, target_index_holder, target_in_wheel] = gen_drop_letterblock_two(wheel_matrix{i}, target_letter{j}, target_per_block, oddball_drops);
    end
    if target_in_wheel
        target_wheel_index = i;
        target_index = target_index_holder;
    end
end

 end

