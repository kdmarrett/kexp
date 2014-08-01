function [  IWI, tot_trial, tot_wheel, letter_difference, min_wheel, preblock, ILI, tot_wav_time ] = assignTimeVars( wheel_matrix_info, fs, tot_cyc, letter_samples, token_rate_modulation, preblock_prime_sec, postblock_sec, ILIms, token_rates )

	% CONVERT TO SAMPLES
    % cycle_sample = ceil(cycle_time * fs);
    postblock = ceil(postblock_sec * fs);  
    preblock = ceil(preblock_prime_sec * fs);
    if token_rate_modulation
    	for i = 1:length(wheel_matrix_info)
    		ILIsec = 1 / (token_rates(i));
    		ILI(i) = ceil(ILIsec * fs);
    	end
    else
	ILI = repmat( (ceil((ILIms ./ 1000) .* fs)), length(wheel_matrix_info), 1);
	end

    % ASSIGN INTER-LETTER-INTERVAL (ILI) FOR EACH GROUP
 %    for i = 1:length(wheel_matrix_info)
	%     if token_rate_modulation
	%         ILI(i) = ceil(cycle_sample / wheel_matrix_info(i)); %INTER-LETTER-TIME determined by each wheel
	%     else
	%         ILI(i) = ceil(cycle_sample / wheel_matrix_info(1)); %INTER-LETTER-TIME determined by first wheel
	%     end
	% end
	% rough_tot_wheel = cycle_sample * tot_cyc;

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

	for i = 1:length(wheel_matrix_info)
	    	% tot_cycle_sample(i) = wheel_matrix_info(i) * ILI(i) + letter_samples;
	    	% spill_over(i) = tot_cycle_sample(i) - cycle_sample;
	    	% adjusted_spill_over(i) = spill_over(i) + IWI * (i - 1); % spillover with respect to wheel onset time
	    	tot_wheel(i) = 1 + ((tot_cyc * wheel_matrix_info(i) - 1) * ILI(i)) + letter_samples; % final calc. of wheel length
	    	if token_rate_modulation
	    		adjusted_wheel(i) = tot_wheel(i) + IWI * (i - 1);
	    	end
	end
	
	% FIND SPECIFIC END TIMES FOR EACH WHEEL AND ENTIRE TRIAL
	if token_rate_modulation
		% trial_spill_over = max(adjusted_spill_over); % finds the spillover relative to the trial ambivalent to wheel spillover
		% tot_response_section = rough_tot_wheel + trial_spill_over;
		tot_response_section = max(adjusted_wheel);
	else 
	    	tot_response_section = tot_wheel(length(wheel_matrix_info)) + IWI * 2; 
	end
	%TOTAL SAMPLES IN EACH TRIAL OF CYCLES
	tot_trial = ceil(preblock + tot_response_section + postblock);
	tot_wav_time = tot_trial / fs;
end


