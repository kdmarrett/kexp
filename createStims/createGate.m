function [output] = createGate(input, fs, start_gate, end_gate)
% createGate takes sound files and smooths the ends of the files to reduce popping
% where sounds have been cut off.  
% start_gate is a boolean that smooths the beginning of the file
% end_gate smooths the end

% DESIGN BASIC AMPLITUDE ENVELOPE
gateDur = .03; % duration of the gate in seconds
gate = cos(linspace(0, pi, floor(fs * gateDur))); % diminish envelope by one half period of cos 
endGate = ((gate + 1) / 2)'; 
begGate = flipud(endGate); %inflection of begGate

[m, n] = size(endGate);
[row, col] = size(input);
sustain = ones((row - 2 * m), 1); % leave inner section

% INCORPORATE BOOLS
if ~start_gate
	begGate = ones(m, n);
end
if ~end_gate
	endGate = ones(m, n);
end

% COMBINE FINAL ENVELOPE AROUND INPUT SOUND
envelope = [begGate; sustain; endGate];
if col ~= 1
	envelope = repmat(envelope, 1, col);
end
output = envelope .* input;
end
