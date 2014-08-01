function [ ] = createStruct(fn)
%CREATE STIM FILE STRUCTURE IF DOES NOT ALREADY EXIST
    if ~exist(fn, 'dir')
        mkdir(fn)
    end
end

