function [data_cell, models, tmin, tmax, sfreq] = create_models(data_file, mdl_tplate)

% Load the data:
load(data_file)
% Fetch each unique condition:
cond_labels = cellstr(cond_labels);
% Extract each single condition:
cond_u = unique(cond_labels);

% Prepare a cell storing the data for each condition:
data_cell = cell(length(cond_u), 1);
models = cell(length(cond_u), 1);

% Loop through each condition:
for cond_i = 1:length(cond_u)
    % Extract the condition:
    cond = cond_u{cond_i};

    % 1- Extract the data for this condition:
    data_cell{cond_i} = data(strcmp(cond_labels,cond), :);

    % 2- Extract and format the latency:
    eventtimes = latency(strcmp(cond_labels,cond), :);
    eventtimes = unique(eventtimes, 'rows');
    % Sort the events times:
    [sorted_evt_times, I] = sort(eventtimes);
    eventlabels = cellstr(eventlabels);
    eventlabels = eventlabels(I);
    % Correct the latencies to match the sampling rate:
    sorted_evt_times = ((round(sorted_evt_times .* sfreq) .* 1/sfreq)) .* 1000;

    % 3- Extract and format the box parameters:
    cond_boxes = box_times(strcmp(cond_labels,cond), :);
    % For column with multiple values, compute the median (RT data):
    for bx_i = 1:size(cond_boxes, 2)
        if length(unique(cond_boxes(:, bx_i))) > 1
            cond_boxes(:, bx_i) = nanmedian(cond_boxes(:, bx_i));
        end
    end
    % Extract unique values:
    cond_boxes = unique(cond_boxes, 'rows');
    % Convert to an n by 2 matrix:
    cond_boxes = reshape(cond_boxes, 2, numel(cond_boxes) / 2)';
    % Correct the latencies to match the sampling rate:
    cond_boxes = ((round(cond_boxes .* sfreq) .* 1/sfreq)) .* 1000;
    cond_boxes = mat2cell(cond_boxes, ones(1, numel(cond_boxes) / 2), 2);
    bx_lbls = cellstr(box_labels);

    %----------------------------------------------------------------------
    % Create the pret model for this condition:
    model = mdl_tplate;
    model.samplerate = sfreq;  % Sampling  frequency:
    model.eventtimes = sorted_evt_times;  % Events labels
    model.eventlabels = eventlabels;  % Events labels
    model.boxtimes = cond_boxes;  % Boxes onset and offsets
    model.boxlabels = bx_lbls;  % Labels for each box
    model.ampbounds = repmat(mdl_tplate.ampbounds,1,...
        length(model.eventtimes));  % Repeat the default parameters for each event
    model.latbounds = repmat(mdl_tplate.latbounds,1,...
        length(model.eventtimes));  % Repeat the default parameters for each event
    model.boxampbounds = repmat(mdl_tplate.boxampbounds,1,...
        length(model.boxtimes));  % Repeat the default parameters for each event
    model.cond_name = cond;  % Add the name of the condition for future reference
    models{cond_i} = model;
    
end
end