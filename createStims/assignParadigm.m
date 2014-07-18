function [wheel_matrix_info, possibleLetters, target_letter, rearrangeCycles, tone_constant, ener_mask, letters_used, token_rate_modulation, AM_freq, AM_pow, shiftedLetters, instrNote_shifted, instrNote, envelope_type, letter_fine_structure, ILIms, sync_cycles  ] = assignParadigm(paradigm, letterArray, env_instrNotes)
	% Assign basic design parameters of each paradigm type
    % +++ assign conditions for last several parameters
    ILImsBase = 400;

	% ASSIGN LETTERS PER WHEEL
    if paradigm(1)
        % wheel_matrix_info = [10 9 7]; % token rate changes according to letters in each wheel
        wheel_matrix_info = [10 10 10]; % token rate changes according to letters in each wheel
        token_rate_modulation = 1;  % bool to change token rate
        ILIms = repmat(ILImsBase, 1, 3);
    else 
        % wheel_matrix_info = [9 9 8]; %token rate remains the same in all wheels
        wheel_matrix_info = [10 10 10]; %token rate remains the same in alls wheels
        token_rate_modulation = 0;
        ILIms = repmat(ILImsBase, 1, 3);
    end
    letters_used = sum(wheel_matrix_info);
    
    % ASSIGN LETTER ORDERING
    if paradigm(2)
        possibleLetters = letterArray.displaced;  %displaced order
    else
        possibleLetters =  letterArray.alphabetic; %alphabet order
    end
    
    % % ASSIGN TARGET LETTER
    if paradigm(3)
        target_letter_i = {'B' 'C' 'D' 'E' 'G' 'P' 'T' 'V'}; %  of all letters ending [i] but 'Z'
        target_letter = target_letter_i(randi([1, length(target_letter_i)])); %choose randomly
    else
        target_letter = {'R'};
    end
   
    % ASSIGN INTER-CYCLE ORDERING
    if paradigm(4)
        rearrangeCycles = 1; %must also have maximally displaced letters
    else
        rearrangeCycles = 0;
    end
    
    % ASSIGN TONE CHARACTERISTICS
    if paradigm(5)
        tone_constant = 1; %tones are assigned randomly to letters and tied to letters for trials
    else
        tone_constant = 0;
    end

    %  USE SAME LETTER ACROSS WHEELS AND CYCLES
    if paradigm(6)
        ener_mask = 1;
    else
        ener_mask = 0;
    end

    %  ASSIGN SPEAKERS


    %AMPLITUDE MODULATOR SHIFTS AMPLITUDE
    AM_freq = [0 0 0 0 0 0 0 0]; %Hz rate of amplitude modulator elements for each wheel 0 for none
    AM_pow =  [0 0 0 0 0 0 0 0]; %decibel of each AM for each corresponding wheel

    % ASSIGN PITCH SHIFTING
    shiftedLetters = 0; % bool letters are monotonized and shifted within one octave above and below A3
    assert(~shiftedLetters, 'kdm recordings of pause, read, etc must also be shifted before creating a shifted letter stimuli')
    instrNote_shifted = 1; % instrument tones go above and below A4 then are later combined with raw letter recordings
    instrNote = 1; % bool to include instrument notes
    letter_fine_structure = 1;
    sync_cycles = 0;
    if env_instrNotes
        envelope_type = 'env';
    else
        envelope_type = 'sin';
    end
end
