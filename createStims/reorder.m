function [ output ] = reorder( input )
%UNTITLED2 Summary of this function goes here
%   randomly shuffles the elements of any matrix or array

ind = randperm(length(input));
output = input(ind);

end

