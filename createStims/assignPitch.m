function [pitch_wheel, angle_wheel, total_pitches, list_of_pitches, start_semitone_index ] = assignPitch(wheel_matrix_info, tot_cyc, scale_type, pitches, descend_pitch)

%establishes the angles to use for the varying amount of wheels
baseline_angle{1} = [0];
baseline_angle{2} = [-90, 90];
baseline_angle{3} = [-90, 0,  90];
baseline_angle{4} = [-90, -30, 30, 90];
baseline_angle{5} = [-90, -40, 0, 40, 90];

wheel_num = length(wheel_matrix_info);
angles = baseline_angle{wheel_num};

%starting note of diatonics
% 3 wheel D, G, A  (semitones: -7.0 , -2.0, 0)
% 4 wheel D, C, G, A (semitones: -7.0, -9.0, -2.0, 0)
if strcmpi(scale_type, 'diatonic')
    if wheel_num == 3
        start_semitone_index = [3 8 10];
    elseif wheel_num == 4 
        start_semitone_index = [ 3 1 8 10];
    else
        fprintf('Error: need to define semitones for this number of wheels')
    end
else if strcmpi(scale_type, 'whole')
    if wheel_num == 3
        start_semitone_index = [1 5 1];
        % start_semitone_index = [1 7 1];
    else
        fprintf('Error: need to define semitones for this number of wheels')
    end
end

for i = 1:wheel_num
    temp = [];
    letters_wheel = wheel_matrix_info(i);
    angle_wheel{i} = repmat(angles(i), tot_cyc, letters_wheel);
    if strcmpi(scale_type, 'diatonic')
        temp = pitches.diatonic((start_semitone_index(i)):(start_semitone_index(i) + letters_wheel - 1));
        total_pitches = length(pitches.diatonic);
        list_of_pitches = pitches.diatonic;
    else      
        % temp = pitches.whole(i:(i+letters_wheel - 1));
        temp = pitches.whole(start_semitone_index(i) :(start_semitone_index(i) + letters_wheel - 1));
        total_pitches = length(pitches.whole);
        list_of_pitches = pitches.whole;
    end
    if descend_pitch(i)
        temp = fliplr(temp) ;
    end
    pitch_wheel{i} =  repmat(temp, tot_cyc, 1);
end
end

