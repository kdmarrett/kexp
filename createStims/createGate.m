function [output] = createGate(input, fs, start_gate, end_gate)
% createGate takes sound files and smooths the ends of the files to reduce popping
% where sounds have been cut off.  
% start_gate is a boolean that smooths the beginning of the file
% end_gate smooths the end

% DESIGN BASIC AMPLITUDE ENVELOPE
gateDur = .03; % duration of the gate in seconds
gate = cos(linspace(2 * pi, 3 * pi, fs * gateDur)); % diminish envelope by half one period of sine
endGate = ((gate + 1) / 2)'; 
begGate = flipud(endGate); %inflection of begGate

[m, n] = size(endGate);
sustain = ones((length(input) - 2 * m), 1); % leave inner section

% INCORPORATE BOOLS
if ~start_gate
	begGate = ones(m, n);
end
if ~end_gate
	endGate = ones(m, n);
end

% COMBINE FINAL ENVELOPE AROUND INPUT SOUND
envelope = [begGate; sustain; endGate];
output = envelope .* input;

end
