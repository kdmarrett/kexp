function [output] = swap(input, a, b)
%helper function for cherry picking trial ordering

a = find(input == a);
b = find(input == b);
temp = input(a);
input(a)= input(b);
input(b) = temp;

output = input;
