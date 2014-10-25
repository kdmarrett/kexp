function [  IWI, tot_trial, tot_wheel, letter_difference, min_wheel, preblock, ILI, tot_wav_time, min_wheel_time, min_wheel_time_ind ] = assignTimeVars( wheel_matrix_info, fs, tot_cyc, letter_samples, token_rate_modulation, preblock_prime_sec, postblock_sec, ILIms, token_rates )

	% CONVERT TO SAMPLES
    postblock = ceil(postblock_sec * fs);  
    preblock = ceil(preblock_prime_sec * fs);

    % ASSIGN INTER-LETTER-INTERVAL (ILI) FOR EACH GROUP FOR TOKEN-RATE MODULATED CONDITION
    if token_rate_modulation
    	for i = 1:length(wheel_matrix_info)
    		ILIsec = 1 / (token_rates(i));
    		ILI(i) = ceil(ILIsec * fs);
    	end
    else
		ILI = repmat( (ceil((ILIms ./ 1000) .* fs)), length(wheel_matrix_info), 1);
	end

	% ADJUST IWI FOR TOKEN-RATE MODULATED CONDITION
	if token_rate_modulation 
	    [y, ind] = max(wheel_matrix_info);
	    IWI = ceil((2 * ILI(ind)) / 3); % IWI timing between letters played from one wheel to the next
	    letter_difference = [];
	    min_wheel  =[];
	else
	    IWI = ceil(ILI(1) / length(wheel_matrix_info));
	    [minimum, min_wheel] = min(wheel_matrix_info);
	    letter_difference = max(wheel_matrix_info) - minimum; %used to sync wheels if specified by sync_wheel
	end

	% FIND SPECIFIC LENGTHS FOR EACH WHEEL
	for i = 1:length(wheel_matrix_info)
    	% tot_cycle_sample(i) = wheel_matrix_info(i) * ILI(i) + letter_samples;
    	% spill_over(i) = tot_cycle_sample(i) - cycle_sample;
    	% adjusted_spill_over(i) = spill_over(i) + IWI * (i - 1); % spillover with respect to wheel onset time
    	tot_wheel(i) = 1 + ((tot_cyc * wheel_matrix_info(i) - 1) * ILI(i)) + letter_samples; % final calc. of individual wheel length
    	if token_rate_modulation
    		adjusted_wheel(i) = tot_wheel(i) + IWI * (i - 1);  % wheel length after accounting for start time
    	end
	end
	
	% [max_wheel_time, max_wheel_time_ind] = max(adjusted_wheel)
	% [min_wheel_time, min_wheel_time_ind] = min(adjusted_wheel)
	% letters_in_trial = zeros((length(wheel_matrix_info) - 1), 1)
	% index = 1
	% for i = 1:length(wheel_matrix_info)
	% 	if i ~= max_wheel_time_ind
	% 		floor(max_wheel_time / ILI(i))
	% 		index = index + 1;
	% end

	% FIND SPECIFIC END TIMES FOR ENTIRE RESPONSE SECTION OF TRIAL
	if token_rate_modulation
		% trial_spill_over = max(adjusted_spill_over); % finds the spillover relative to the trial ambivalent to wheel spillover
		% tot_response_section = rough_tot_wheel + trial_spill_over;
		tot_response_section = max(adjusted_wheel);
		[min_wheel_time, min_wheel_time_ind] = min(adjusted_wheel);
	else 
    	tot_response_section = tot_wheel(length(wheel_matrix_info)) + IWI * 2; % This should equal max(adjusted_wheel)
		min_wheel_time = [];
		min_wheel_time_ind = [];
	end

	%TOTAL SAMPLES IN EACH TRIAL OF CYCLES
	tot_trial = ceil(preblock + tot_response_section + postblock);
	tot_wav_time = tot_trial / fs;

	% CUT WHEEL TRACKS TO MATCH IN LENGTH
	% if token_rate_modulation
	% 	if j ~= min_wheel_time_ind
	% 		final_wheel = trimSoundVector(final_wheel, fs, min_wheel_time, 1, 1)
	% 	end
	% end
end


