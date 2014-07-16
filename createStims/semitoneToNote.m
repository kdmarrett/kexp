function [note] = semitoneToNote( semitone, pitches )
	index = find(strcmp(semitone, pitches.all));
	note = pitches.notes{index};
end