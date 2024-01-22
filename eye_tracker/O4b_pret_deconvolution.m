% housekeeping:
clear all
close all
% Addpath to the pret toolbox:
addpath("C:\Users\alexander.lepauvre\Documents\GitHub\PRET")

%% Set parameters:
% Set the data parameters:
root = fullfile(extractBidsRoot("..\environment_variables.py"), 'derivatives', 'pret');
session = '1';
task = 'prp';
subjects = ["SX102", "SX103", "SX105", "SX106", "SX107", "SX108", "SX109", "SX110", "SX112", "SX113", ...
    "SX114", "SX115", "SX116", "SX119", "SX120", "SX121"]; % List of subjects to model , "SX118"

% PRET parameters:
% Preprocessing parameters:
baseline_win = [-0.2  * 1000, 0  * 1000]; % Baseline time window for baseline normalization
normflag = 1;
blinkflag = 0;
% Model parameters:
model_window = [0, 2500]; % Time window to model
yintflag = 0;
slopeflag = 0;
ampbounds = [-100; 100]; % Amplitude can vary between positive and negative values
latbounds = [0; 500]; % Allow latencies to vary to 500ms after the event of interest
boxampbounds = [0;100]; % Box amplitude constrained to positive values, as it models cognitive load of the task
tmaxbounds = [500;2500]; % Bound for the tmax parameter of the pupil response function
yintval = 0;
slopeval = 0;
optimnum = 8; % Take the 20 best starting values for optimization
wnum = 2; % Number of cores to use

% Prepare variable to store the results
res = [];
%% Subject loop:
for subject_i = 1:length(subjects)
    subject = subjects{subject_i};
    fprintf("Modelling the data of subject: %s\n", subject)

    % Load the data:
    load(fullfile(root, sprintf("sub-%s_ses-%s_task-%s_epochs.mat", subject, session, task)));

    % Fetch each unique condition:
    cond_labels = cellstr(cond_labels);
    cond_u = unique(cond_labels);
    sj_res = {};

    % Loop through each experimental condition:
    for cond_i = 1:length(cond_u)
        cond = cond_u{cond_i};
        fprintf("Modelling condition: %s\n", cond)
        % Extract the data of this condition:
        cond_data = {data(strcmp(cond_labels,cond), :)};
        % Set the reaction time to the mean:
        eventtimes = latency(strcmp(cond_labels,cond), :);
        eventtimes(:, 3) = nanmean(eventtimes(:, 3));
        % Extract  the latency of this condition:
        eventtimes = unique(eventtimes, 'rows');

        %% Create preprocessing options;
        dflt_opt = pret_default_options;
        % Preprocessing:
        prepro_opt = dflt_opt.pret_preprocess;
        prepro_opt.normflag = normflag;
        prepro_opt.blinkflag = blinkflag;

        %% create model for the data
        % Pretending that we are naive to the model that we used to create our
        % data, let's create a model to actually fit the data to.
        model = pret_model();

        % While the trial window of our task is from -500 to 3500 ms, here we are
        % not interested in what's happening before 0. So let's set the model 
        % window to fit only to the region betweeen 0 and 3500ms (the cost 
        % function will only be evaluated along this interval).
        model.window = model_window;

        % We already know the sampling frequency.
        model.samplerate = sfreq;

        % Add the events time of the task. Importantly, it needs to be
        % sorted:
        rt = (round(eventtimes(3) * sfreq) * 1 / sfreq) * 1000;
        [sorted_evt_times, I] = sort(eventtimes); 
        model.eventtimes = round(sorted_evt_times .* 1000);
        model.eventlabels = cellstr(eventlabels); %optional
        model.eventlabels = model.eventlabels(I);
        model.boxtimes = {[0 rt]};
        model.boxlabels = {'task'}; %optional

        % Let's say we want to fit a model with the following parameters:
        % event-related, amplitude, latency, task-related (box) amplitude,
        % and the tmax of the pupil response function. We turn the other parameters
        % off.
        model.yintflag = yintflag;
        model.slopeflag = slopeflag;

        % Now let's define the bounds for the parameters we decided to fit. We do
        % not have to give values for the y-intercept and slope because we are not
        % fitting them.
        model.ampbounds = repmat(ampbounds,1,length(model.eventtimes));
        model.latbounds = repmat(latbounds,1,length(model.eventtimes));
        model.boxampbounds = boxampbounds;
        model.tmaxbounds = tmaxbounds;

        % We need to fill in the values for the y-intercept and slope since we will
        % not be fitting them as parameters.
        model.yintval = yintval;
        model.slopeval = slopeval;

        %% Set the options for the estimation:
        set_opt = pret_estimate_sj();
        set_opt.pret_estimate.optimnum = optimnum;
        %% Preprocessing:
        sj = pret_preprocess(cond_data,sfreq,[tmin * 1000, tmax  * 1000], {cond}, baseline_win,prepro_opt);

        %% Fit the model:
        tstart = tic;
        sj = pret_estimate_sj(sj,model,wnum,set_opt);
        tend = toc(tstart);
        fprintf("Elapsed time: %d\n", tend)
        sj.subject = subject;
        sj.estim.(cond).eventlabels = model.eventlabels;
        sj_res = [sj_res, sj];
    end
    res = [res, {sj_res}];
end

%% Save the results:
results = [];
condition_columns = ["SOA", "duration", "lock", "task"];
for sub_i = 1:length(res)
    for cond_i = 1:length(res{sub_i})
        T = table();
        T.subject = res{sub_i}{cond_i}.subject;
        condition = cellstr(res{sub_i}{cond_i}.conditions{1});
        condition_split = split(condition{1}, '_');
        num_conditions = numel(condition_split);
        for i = 1:num_conditions
            T.(condition_columns{i}) = condition_split(i);
        end
        latencies = res{sub_i}{cond_i}.estim.(condition{1}).latvals;
        eventlabels = res{sub_i}{cond_i}.estim.(condition{1}).eventlabels;
        [eventlabels, I] = sort(eventlabels);
        latencies = latencies(I);
        num_latencies = numel(latencies);
        for i = 1:num_latencies
            T.(eventlabels{i}) = latencies(i);
        end
        results = [results; T];
    end
end

writetable(results, fullfile(root, sprintf("ses-%s_task-%s_desc-deconvolution_res.csv", session, task)))
