function [wheel_matrix_info, possibleLetters, target_letter, rearrangeCycles, tone_constant] = assignParadigm(paradigm)
	% Assign basic design paraeters of each paradigm type

	% ASSIGN LETTERS PER WHEEL
    if paradigm(1)
        wheel_matrix_info = [9 8 7]; 
    else
        wheel_matrix_info = [8 8 8];
    end
    
    % ASSIGN LETTER ORDERING
    if paradigm(2)
        possibleLetters = letterArray.displaced;  %displaced order
    else
        possibleLetters =  letterArray.alphabetic; %alphabet order
    end
    
    % ASSIGN LETTER BLOCK
    if paradigm(3)
        target_letter_i = {'B' 'C' 'D' 'E' 'G' 'P' 'T' 'V' 'Z'}; %  of all letters ending [i]
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
