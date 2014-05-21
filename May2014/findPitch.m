function [ letter_pitch ] = findPitch(letterArray, letter_to_pitch, list_of_pitches, letter )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


letter_pitch = list_of_pitches{find(sum(strcmp(letter, letter_to_pitch)))};

end

