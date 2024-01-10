function bidsRoot = extractBidsRoot(filePath)
    % Open the file
    fileID = fopen(filePath, 'r');
    if fileID == -1
        error('Unable to open file');
    end

    % Read the contents as a string
    fileContent = fscanf(fileID, '%c');
    fclose(fileID); % Close the file

    % Find the position of 'bids_root = '
    startIndex = strfind(fileContent, 'bids_root = ');
    if ~isempty(startIndex)
        % Extract substring after 'bids_root = '
        bidsSubstring = fileContent(startIndex + length('bids_root = '):end);

        % Find the end position of bids_root value
        endIndex = strfind(bidsSubstring, 'raw_root') - 2; % -2 to exclude the closing quote and newline

        % Extract bids_root value
        bidsRoot = strtrim(bidsSubstring(1:endIndex));

        % Remove leading 'r' and quotes if present
        if bidsRoot(1) == 'r' && (bidsRoot(2) == "'" || bidsRoot(2) == '"')
            bidsRoot = bidsRoot(3:end-1);
        end
    else
        error('bids_root not found');
    end
end