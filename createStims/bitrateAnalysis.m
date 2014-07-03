% bitrateAnalysis script April 27th update
% Detailed explanation goes here
clear all
close all

target_dope =  3;
non_target_dope = 2;
letters_wheel = [5 6 6 8]';
wheel_num = [5 5 4 3]';
inter_wheel_msec = [40 50 60 70 80 90 100];
inter_wheel_sec = inter_wheel_msec ./ 1000;
selections = zeros(4, 1);

for i = 1:length(inter_wheel_sec) %looping by column through the different inter_wheel possible times
    for j = 1:length(wheel_num) % looping by row for each trial type
        inter_letter_sec(j, i) = wheel_num(j) * inter_wheel_sec(i);
        total_letters_wheel(j) = (letters_wheel(j) - 1) * non_target_dope + target_dope;
        l_tar_P(j) = target_dope / total_letters_wheel(j); %local(intra_wheel) probability of target occurring
        t_tar_P(j) = target_dope / (total_letters_wheel(j) * wheel_num(j)); %probability of target occurring across entire trial
        trial_time_s(j, i) = (inter_wheel_sec(i) * (wheel_num(j) -1)) + inter_letter_sec(j, i) * (total_letters_wheel(j));
        selections(j) = wheel_num(j) * letters_wheel(j);
    end
end

trial_time_m = trial_time_s ./ 60;


figure(1)
bar3(inter_letter_sec)
grid on
rotate3d on
xlabel('Temporal offset between wheels')
ylabel('Condition Type')
zlabel('Temporal offset between intra wheel letters (s)');

figure(2)
bar3(trial_time_s)
grid on
rotate3d on
title('Trial time')
xlabel('letters per wheel')
ylabel('Condition Type')
zlabel('Time (s)')
% zlabel('time of full letter sweep (ms)')

P = [.80 .90 .95]; %probability/accuracy of user
inter_wheel_index = 5; %index of inter_wheel time to use for bit/minute analysis
for i = 1:length(wheel_num) %create rows for each trial type
    for j = 1:length(P)  % create columns for each probability/accuracy
        N = selections(i); %possible selections in each trial type
        B(i, j) = log2(N) + P(j) * log2(P(j)) + (1 - P(j)) * log2((1 - P(j))/(N - 1)); % total bits per trial type
        B_rate(i, j) = B(i, j) ./ trial_time_m(i, inter_wheel_index);
    end
end

condition_index = 4;
for i = 1:length(inter_wheel_sec) %create rows for each trial type
    for j = 1:length(P)  % create columns for each probability/accuracy
        N = selections(condition_index); %possible selections in each trial type
        B_acrossIWI(i, j) = log2(N) + P(j) * log2(P(j)) + (1 - P(j)) * log2((1 - P(j))/(N - 1)); % total bits per trial type
        B_rate_acrossIWI(i, j) = B_acrossIWI(i, j) ./ trial_time_m(condition_index, i);
    end
end

figure(3)
bar3(B)
title('Bits per trial across condition type and user accuracy')
xlabel('Probability/Accuracy of User')
ylabel('Condition Type')
zlabel('Bit/min')
grid on
rotate3d on

figure(4)
bar3(B_rate)
title('Bit/min per trial across condition type and user accuracy')
xlabel('Probability/Accuracy of User')
ylabel('Condition Type')
zlabel('Bit/min')
grid on
rotate3d on

figure(5)
bar(t_tar_P)
title('Probability of target across whole trial')
xlabel('Condition')
ylabel('Probability')

figure(6)
bar(l_tar_P)
title('Local probability of target within target wheel')
xlabel('Condition')
ylabel('Probability')

figure(7)
bar3([selections, 7* B(:, 2), B_rate(:, 2)])
title('Selections compared to bit and bitrate across trial conditions')
ylabel('Condition')
xlabel('Selections,   7* bits,   Bit rate')

%%Narrow down ideal parameters for wheel and letters per wheel parameters
% test trial variables named _t
wheel_num_t = [1:12];
letters_wheel_t = [1:8];
inter_wheel_sec_t = .100;
original_dope = non_target_dope;
P_index = 2;  %probability of 90
thresholdTest = 0; %boolean
thresh = 26; %must contain 26 letters
normLocalProb = 1; %boolean
probThresh = .3;
for i = 1:length(wheel_num_t)
    for j = 1:length(letters_wheel_t)
        inter_letter_sec_t(i, j) = wheel_num_t(i) * inter_wheel_sec_t;
        total_letters_wheel(i, j) = (letters_wheel_t(j) - 1) * non_target_dope + target_dope;
        l_tar_P_t(i, j) = target_dope / total_letters_wheel(i, j) %local(intra_wheel) probability of target occurring
        if normLocalProb
            while ((l_tar_P_t(i, j) > probThresh) & (l_tar_P_t(i, j) ~= 1)),
                non_target_dope = non_target_dope + 1;
                total_letters_wheel(i, j) = (letters_wheel_t(j) - 1) * non_target_dope + target_dope;
                l_tar_P_t(i, j) = target_dope / total_letters_wheel(i, j) %local(intra_wheel) probability of target occurring
            end
            dope_matrix(i, j) = non_target_dope;
            non_target_dope  = original_dope;
        end
        t_tar_P_t(i, j) = target_dope / (total_letters_wheel(j) * wheel_num_t(i)); %probability of target occurring across entire trial
        trial_time_s(i, j) = (inter_wheel_sec_t * (wheel_num_t(i) -1)) + inter_letter_sec_t(i, j) * (total_letters_wheel(i, j));
        trial_time_m(i, j) = (trial_time_s(i, j) / 60);
        selections_t(i, j) = wheel_num_t(i) * letters_wheel_t(j);
        N = selections_t(i, j); %possible selections in each trial type
        if thresholdTest
            if N >= thresh
                B_t(i, j) = log2(N) + P(P_index) * log2(P(P_index)) + (1 - P(P_index)) * log2((1 - P(P_index))/(N - 1)); % total bits per trial type
                B_rate_t(i, j) = B_t(i, j) ./ trial_time_m(i, j);
            else
                B_t(i, j) = 0;
                B_rate_t(i, j) = 0;
            end
        else
            B_t(i, j) = log2(N) + P(P_index) * log2(P(P_index)) + (1 - P(P_index)) * log2((1 - P(P_index))/(N - 1)); % total bits per trial type
            B_rate_t(i, j) = B_t(i, j) ./ trial_time_m(i, j);
        end
    end
end

figure(8)
bar3(selections_t)
title('Selections across wheel number and letter per wheel');
xlabel('letters per wheel]')
ylabel('Wheel num')
rotate3d on

figure(9)
bar3(B_rate_t)
% title('Bit rate across wheel number and letter per wheel');
% xlabel('letters per wheel')
% ylabel('Wheel num')
rotate3d on
% 
% figure(10)
% bar3(trial_time_s)
% title('trial time (s) across wheel number and letter per wheel');
% xlabel('letters per wheel')
% ylabel('Wheel num')
% rotate3d on
% 
% figure(11)
% bar3(l_tar_P_t)
% title('local intrawheel probability of target letter');
% xlabel('letters per wheel')
% ylabel('Wheel num')
% rotate3d on
% 
% figure(12)
% bar3(t_tar_P_t)
% title('global trial probability of letter');
% xlabel('letters per wheel')
% ylabel('Wheel num')
% rotate3d on
% 
% 
