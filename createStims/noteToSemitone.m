function [semitone] = noteToSemitone( note, pitches )
	index = find(strcmp(note, pitches.notes));
	semitone = pitches.all{index};
end
