function [ ] = createStruct(fn)
%CREATE STIM FILE STRUCTURE IF DOES NOT ALREADY EXIST
    if exist(fn, 'dir')
    else
        mkdir(fn)
    end
end

