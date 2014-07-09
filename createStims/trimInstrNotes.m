function [] = trimInstrNotes(fs, instrnote_path, letter_samples, pitches.notes, instrument_dynamics)

x = linspace(0, pi, letter_samples)
envelope = sin(x)
for i = 1:length(pitches.notes)
	[pitches.notes{i}, fs] = wavread(instrnote_path);
	sin(x)  =