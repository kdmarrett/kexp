function [possibleLetters, rearrangeCycles, ...
tone_constant, ener_mask, letters_used, token_rate_modulation,...  
AM_freq, AM_pow, shiftedLetters, instrNote_shifted, instrNote, ...
envelope_type, letter_fine_structure, sync_cycles, id_code  ] = ...  
assignParadigm(paradigm, letterArray, env_instrNotes, total_letters, ...  
wheel_matrix_info )

	% assign basic design parameters of each paradigm type booleans for
	% design features, ordered: control, each wheel group at an
	% orthogonal frequency of token rate, maximally displaced ordering,
	% target letter 'r' as opposed to x[i],
	% letter orders are retained across cycles, tone is assigned
	% randomly as opposed to contiguously, each wheel group given a
	% unique frequency am, each group given the same am frequency 

    ener_mask = 0;% assigns all non target letters to letter O 
	target_letter = 'R'; %dummy target letter

    % ASSIGN LETTERS PER WHEEL
    if paradigm(1)
        token_rate_modulation = 1;  % bool to change token rate
    else 
        token_rate_modulation = 0;
    end

    % TEST
    letters_used = sum(wheel_matrix_info); 
    if ~(letters_used == total_letters)
        fprintf('Error: not all letters used')
    end
    
    % ASSIGN LETTER ORDERING
    if paradigm(2)
        possibleLetters = letterArray.displaced;  %displaced order
    else
        possibleLetters =  letterArray.alphabetic; %alphabet order
    end
    
    % % ASSIGN TARGET LETTER
    %if paradigm(3)
        %target_letter_i = {'B' 'C' 'D' 'E' 'G' 'P' 'T' 'V'}; %  of all letters ending [i] but 'Z'
        %target_letter = target_letter_i(randi([1, length(target_letter_i)])); %choose randomly
    %else
        %target_letter = {'R'};
    %end
   
    % ASSIGN INTER-CYCLE ORDERING
    if paradigm(4)
        rearrangeCycles = 1; %must also have maximally displaced letters
    else
        rearrangeCycles = 0;
    end

    %  Assign id code for stamping
    if paradigm(2)
        if paradigm(4)
            id_code = de2bi(3, 2);
        else
            id_code = de2bi(2, 2);
        end
    else
        id_code = de2bi(1, 2);
    end
    
    % ASSIGN TONE CHARACTERISTICS
    if paradigm(5)
        tone_constant = 1; %tones are assigned randomly to letters and tied to letters for trials
    else
        tone_constant = 0;
    end

    %AMPLITUDE MODULATOR SHIFTS AMPLITUDE
	if paradigm(6)
		AM_freq = [2 11 13 0 0 0 0 0]; %Hz rate of amplitude modulator elements for each wheel 0 for none
		AM_pow =  [.15 .15 .15 0 0 0 0 0]; %decibel of each AM for each corresponding wheel
	elseif paradigm(7)
	   AM_freq = [11 11 11 0 0 0 0 0]; %Hz rate of amplitude modulator elements for each wheel 0 for none
		AM_pow =  [.15 .15 .15 0 0 0 0 0]; %decibel of each AM for each corresponding wheel
	else
	  AM_freq = [0 0 0 0 0 0 0 0];  %Hz rate of amplitude modulator elements for each wheel 0 for none
		AM_pow =  [0 0 0 0 0 0 0 0]; %decibel of each AM for each corresponding wheel
	end

    % ASSIGN PITCH SHIFTING
    shiftedLetters = 0; % bool letters are monotonized and shifted within one octave above and below A3
    assert(~shiftedLetters, 'kdm recordings of pause, read, etc must also be shifted before creating a shifted letter stimuli')
    instrNote_shifted = 1; % instrument tones go above and below A4 then are later combined with raw letter recordings
    instrNote = 0; % bool to include instrument notes
    letter_fine_structure = 1;
    sync_cycles = 0;
    if env_instrNotes
        envelope_type = 'env';
    else
        envelope_type = 'sin';
    end
end
