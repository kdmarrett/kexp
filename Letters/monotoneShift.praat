###########################################
# Silence Trimming and Pitch Manipulation #
# of all sound utterances in a directory  #
#    3 X8 update for home computer                                     
#
# Jing Xia, jxia@cns.bu.edu               #
# Modified by Eric Larson, larsoner@uw.edu#
# Modified again for the Willy experiments by Ross Maddox, rkmaddox@uw.edu#
# Modified again for Karl, Mark experiments by Karl Marrett, kdmarrett@gmail.com#
###########################################

 
# A3
f0 = 220
nfreqs = 25
semis [1]  = -12
semis [2]  = -11
semis [3]  = -10
semis [4]  = -9
semis [5]  = -8
semis [6]  = -7
semis [7]  = -6
semis [8]  = -5
semis [9]  = -4
semis [10] = -3
semis [11] = -2
semis [12] = -1
semis [13] = 0
semis [14] = 1
semis [15] = 2
semis [16] = 3
semis [17] = 4
semis [18] = 5
semis [19] = 6
semis [20] = 7
semis [21] = 8
semis [22] = 9
semis [23] = 10
semis [24] = 11
semis [25] = 12

trimsilences = 1
shiftSound = 0
# RKM already trimmed silences manually

for ifreq from 1 to nfreqs
	tempTone = semis [ifreq]
	freqs [ifreq]  = f0 * 2 ^ (tempTone / 12)
endfor

form Preprocessing WAV files 
	comment Directory of input sound files
	 text sound_directory /Users/nancygrulke/git/kexp/Letters/rawLetters/fec0
	comment Directory of output sound files
	 text out_directory /Users/nancygrulke/git/kexp/Letters/shiftedLetters/fec

0
endform

Create Strings as file list... list 'sound_directory$'/*.wav
numberOfFiles = Get number of strings

for ifile to numberOfFiles
	select Strings list
   	fileName$ = Get string... ifile
   	Read from file... 'sound_directory$'/'fileName$'

        # get the name of the sound
        name$ = selected$("Sound")
	sr = Get sample rate

	if trimsilences
        	###############################################
	        # trimming the start and end silence  
        	###############################################
        
	   	# delimit and mark silence intervals
		# To TextGrid (silences)... 
		# Silence threshold (dB); Minimum silent interval duration (s); Minimum sounding interval duration (s); 
		# Silent interval label; Sounding interval label 
		To TextGrid (silences)... 50 0 -25 0.1 0.1 silent sounding

        	# get the label of fist interval
		# Get label of interval...
		# start interval
		# end interval
		first_label$ = Get label of interval... 1 1

		# relabel the first interval if it is silent
		if first_label$ = "silent"
			Set interval text... 1 1 junk
		endif

		# extract sounding intervals into one Sound object per interval 
        	# while both signal and text grid are selected
		select Sound 'name$'
		plus TextGrid 'name$'
		Extract all intervals... 1 no

        	# name of the 1st interval (1st selected sound from the top)
	        # name of the last interval (1st selected sound from the bottom)

		first_name$ = selected$ ("Sound",1)
		last_name$ = selected$ ("Sound", -1)

		# cut the starting and ending silence
		if first_name$ = "junk"
			minus Sound 'first_name$'
		endif
		if last_name$ = "silent"
			minus Sound 'last_name$'
		endif

        	# concatenate all interval sound
		Concatenate

		select all;
		numberOfSelectedSounds = numberOfSelected ("Sound")

		# remove temporary objects
		for i from 2 to 'numberOfSelectedSounds'
			sound'i' = selected ("Sound", -i)
		endfor
		for i from 2 to 'numberOfSelectedSounds'
			select sound'i'
			Remove
		endfor

		select TextGrid 'name$'
		Remove

		select Sound chain
	endif

        ###############################################
        # changing pitch to disired levels 
        ###############################################

	Rename... insound
	tmin = Get starting time
	tmax = Get finishing time
	dur = tmax-tmin

	for ifreq from 1 to nfreqs
		# create a Sound as the sum of a number of sine waves with equidistant frequencies
		# Create Sound from tone complex...
		# Name: the name of the resulting Sound object. 
		# Start/End time, (s): the time domain of the resulting Sound. 
		# Sampling frequency (Hz) 
		# Phase: determines whether result is sum of sines or cosines
		# Frequency step (Hz): the distance betw. components (fund freq)
	            # First frequency (Hz): lowest freqy component 0:frequencyStep
	  	# Ceiling (Hz): frequency above which no comps are used 0: Nyquist frequency
		# Number of components: how many sinusoids are used 0: the maximum number of components is used, limited by ceiling
		tempFreq = freqs [ifreq]
		tempSemi = semis [ifreq]
		Create Sound from tone complex... toneComplex 0 dur sr Cosine tempFreq 0 0 2
		createDirectory (out_directory$)
		wdir$ = out_directory$ + "/" + fixed$(tempSemi,1) 
		createDirectory (wdir$)
		wfile$ = wdir$ + "/" + fileName$

		# extract a copy of the pitch information 
		# To Manipulation...
		# timeStep: A common value is 0.010 seconds. 
		# min/maximumPitch: A common value is 75/600 Hertz  
		select Sound toneComplex
		To Manipulation... 0.001 50 600
		Extract pitch tier

		# replace the pitch tier
		# only manipulate the voiced pitch
		select Sound insound
		minFreq = tempFreq - 1
		maxFreq = tempFreq + 1
		To Manipulation... 0.0005 109 441
		select PitchTier toneComplex
		plus Manipulation insound
		Replace pitch tier

	        	# resynthesis
		select Manipulation insound
		Get resynthesis (PSOLA)
		select Sound insound
		Rename... outsound
		Write to WAV file... 'wfile$'

		select Sound toneComplex
		plus Manipulation toneComplex
		plus PitchTier toneComplex
		plus Manipulation insound
		plus Sound outsound
		Remove
	endfor
	#-----------------------------------------
	select Sound insound
	Remove

endfor