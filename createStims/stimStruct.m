function [ ] = stimStruct(base_path, blocks)
%CREATE STIM FILE STRUCTURE
for i = 1:blocks
    fn = fullfile(base_path, strcat('block_', int2str(i)));
    if exist(fn, 'dir')
    else
        mkdir(fn)
    end
end
end

