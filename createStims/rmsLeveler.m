function rmsLeveler
% normalizes a set of files to all have the same RMS amplitude. Target
% amplitude is either (1) matched to that of an existing wav file that you
% specify, or (2) the maximum possible RMS that still avoids clipping.
%
% DEPENDS: cell2csv.m
% http://www.mathworks.com/matlabcentral/fileexchange/4400-cell-array-to-csv-file-cell2csv-m
%
% AUTHOR: Daniel McCloy (drmccloy@uw.edu)
% LICENSED UNDER A CREATIVE COMMONS ATTRIBUTION 3.0 LICENSE: http://creativecommons.org/licenses/by/3.0/

%% load in list of wav file paths
    fprintf('\nSelect WAV files to normalize\n');
    [fileNameArray,fileFolderPath,~] = uigetfile('*.wav','Select WAV files to normalize','MultiSelect','on');
    pathList = cell(size(fileNameArray));
    for k=1:length(fileNameArray)
        % convert the list of stimulus filenames into full file paths
        pathList{k} = strcat(fileFolderPath,fileNameArray{k});
    end

%% read in WAV data and calculate RMS and clipping-safe scale factors
    fprintf('Reading WAV files...\n');

	% preallocate RMS / maxAmp / maxScale vectors & cell array of wav data
    rmsVec = zeros(length(pathList),1);
    maxAmp = zeros(length(pathList),1);
    wavData = cell(size(pathList));

	% assuming all files are the same, take the bitrate from the first file
    [~,bitrate] = wavread(pathList{1});
    
	% Read each file, calculate its RMS ampl & max peak
    for m=1:length(pathList)
        wavData{m} = wavread(pathList{m});
        rmsVec(m) = sqrt(sum(wavData{m}(:).^2)/length(wavData{m}(:)));
        maxAmp(m) = max(abs(wavData{m}(:)));
    end

	% the maximum factor by which we could scale each file without inducing clipping
    maxScale = 0.999 ./ maxAmp;

	% the maximum possible RMS level for each file without inducing clipping
    maxRMS = maxScale .* rmsVec;

%% ask whether we're trying to match a target file or not
    rmsMode = input('\nEnter "1" to match RMS of another WAV file (you will be prompted for that file),\nor enter "2" to RMS level based only on your stimuli (maximize RMS but avoid clipping) >> ');
    if rmsMode == 1
        % load in target WAV file that has the desired target RMS amplitude already
        fprintf('Select target WAV file that already has desired RMS level\n');
        [targetFile,targetFolder,~] = uigetfile('*.wav','Select target WAV file to match RMS to:','MultiSelect','off');
        targetPath = strcat(targetFolder,targetFile);
    
		% Read the target file and calculate its RMS amplitude
        targetWAV = wavread(targetPath);
        targetRMS = sqrt(sum(targetWAV(:).^2)/length(targetWAV(:)));
    
        if targetRMS > min(maxRMS)
		% clipping will happen
            clipVec = maxRMS < targetRMS;
            clipCount = sum(clipVec);
            allowClip = input(sprintf('WARNING: using this RMS target will cause clipping in %d of your files. Continue? [y/n] >> ', clipCount), 's');
        
            if strcmpi(allowClip,'n')
            % the user doesn't want clipping
                preventClip = input('Continue leveling to the max RMS value that still avoids clipping? [y/n] >> ', 's');

                if strcmpi(preventClip,'n')
                % the user gives up
                    error('\nCancelling rmsManager: no files were RMS leveled.\n');
                else
                % the user RMS levels anyway, ignoring the target file
                    fprintf('OK, continuing, but leveling to the max value that prevents clipping.');
            		targetRMS = min(maxRMS);
                end
            
            else
            % the user continues despite clipping, so we temporarily turn off the clipping warning and instead write out a list of the files that got clipped.
                disableWarn = warning('off','MATLAB:wavwrite:dataClipped');
                fprintf('OK, continuing, and writing a list of clipped files to the script folder.\n');
                clipList = cell(clipCount,1);
                n = 1;
                for p=1:length(clipVec)
                    if clipVec(p) == 1
                        clipList{n} = pathList{p};
                        n=n+1;
                    end
                end
                % write out the list of clipped files
                cell2csv('listOfClippedFiles.csv',clipList);
            end
        end
    
	elseif rmsMode ~= 2
	% user didn't pick a viable option
        error('\nYou did not push either 1 or 2\n')
	else
	% rmsMode is "2", meaning there is no target file. Thus we calculate the target RMS as the lowest of the maximum RMS levels (i.e., the only one that is safe for all files).
		targetRMS = min(maxRMS);
    end

%% specify output folder
    fprintf('\nSelect output folder for normalized files:\n');
    outputFolder = uigetdir(cd,'Select output folder for normalized files:');

%% rescale the files and write them out
	for k=1:length(pathList)
		temp = wavData{k}*targetRMS/rmsVec(k);
		outputFile = strcat(outputFolder,'/',fileNameArray{k});
		wavwrite(temp,bitrate,outputFile);
	end

    % write success message
	if ~exist('clipList','var')
			clipList = [];
	end
	fprintf('Done! %d files normalized, %d of them were clipped.\n',length(pathList),length(clipList));

    % re-enable warnings for wavwrite clipping:
	if exist('disableWarn','var')
		warning(disableWarn);
	end
end