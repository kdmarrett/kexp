function [pitch_wheel, angle_wheel, total_pitches, list_of_pitches ] = pitch_angle_wheel(wheel_matrix_info, tot_cyc, scale_type)

%establishes the angles to use for the varying amount of wheels
baseline_angle{1} = [0];
baseline_angle{2} = [-90, 90];
baseline_angle{3} = [-90, 0,  90];
baseline_angle{4} = [-90, -30, 30, 90];
baseline_angle{5} = [-90, -40, 0, 40, 90];
angles = baseline_angle{wheel_num};

%starting note of diatonics
% 3 wheel D, G, A  (semitones: -7.0 , -2.0, 0)
% 4 wheel D, C, G, A (semitones: -7.0, -9.0, -2.0, 0)
if strcmpi(scale_type, 'diatonic')
    if wheel_num == 3
        start_semitone_index = [3 8 10];
    end
    if wheel_num == 4 
        start_semitone_index = [ 3 1 8 10];
    end
end

%established the pitch orders for each wheel of letters
%pent_pitches = {'0', '1.0', '2.0', '4.0', '5.0'};
diatonic_pitches = {'-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0' '-1.0','0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0'};
whole_tone_pitches = {'-9.0', '-7.0', '-5.0', '-3.0', '-1.0', '1.0', '3.0', '5.0', '7.0', '9.0'};  

for i = 1:wheel_num
temp = [];
angle_wheel{i} = repmat(angles(i), tot_cyc, letters_wheel);
if strcmpi(scale_type, 'diatonic')
    temp = diatonic_pitches((start_semitone_index(i)):(start_semitone_index(i) + letters_wheel - 1));
    total_pitches = length(diatonic_pitches);
    list_of_pitches = diatonic_pitches;
else      
    temp = whole_tone_pitches(i:(i+letters_wheel - 1));
    total_pitches = length(whole_tone_pitches);
    list_of_pitches = whole_tone_pitches;
end
    pitch_wheel{i} =  repmat(temp, tot_cyc, 1);
end
end

