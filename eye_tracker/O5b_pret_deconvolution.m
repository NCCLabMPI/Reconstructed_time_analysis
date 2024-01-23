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
subjects = ["SX102"]; % List of subjects to model , "SX118"

% PRET parameters:
% Preprocessing parameters:
dflt_opt = pret_default_options;
baseline_win = [-0.2  * 1000, 0  * 1000]; % Baseline time window for baseline normalization
prepro_opt = dflt_opt.pret_preprocess;
prepro_opt.normflag = 1;
prepro_opt.blinkflag = 0;
% Model parameters:
model_template = pret_model();
model_template.window = [0, 2500]; % Time window to model
model_template.yintflag = 0;
model_template.slopeflag = 0;
model_template.ampbounds = [0; 100]; % Amplitude can vary between positive and negative values
model_template.latbounds = [0; 500]; % Allow latencies to vary to 500ms after the event of interest
model_template.boxampbounds = [0; 100]; % Box amplitude constrained to positive values, as it models cognitive load of the task
model_template.tmaxbounds = [500;2500]; % Bound for the tmax parameter of the pupil response function
model_template.yintval = 0;
model_template.slopeval = 0;
% Estimation options:
set_opt = pret_estimate_sj();
set_opt.pret_estimate.optimnum = 1;
wnum = 1; % Number of cores to use

% Prepare variable to store the results
res = cell(length(subjects), 1);
%% Subject loop:
for subject_i = 1:length(subjects)
    subject = subjects{subject_i};
    fprintf("Modelling the data of subject: %s\n", subject)

    % Create the models and the data for this subject:
    [data, models, tmin, tmax, sfreq] = create_models(fullfile(root, sprintf("sub-%s_ses-%s_task-%s_epochs.mat", subject, session, task)), model_template);
    % Prepare a cell to store the results:
    sj_res = cell(length(data), 1);
    % Loop through each experimental condition:
    for cond_i = 1:length(data)
        % Extract the name of the condition, data and model:
        cond = models{cond_i}.cond_name;
        cond_data = data{cond_i};
        model = models{cond_i};
        % Preprocess the data:
        sj = pret_preprocess({cond_data},sfreq,[tmin * 1000, tmax  * 1000], {cond}, baseline_win,prepro_opt);
        % Fit the model:
        sj = pret_estimate_sj(sj,model,wnum,set_opt);
        % Add information to the model:
        sj.subject = subject;
        sj.estim.(cond).eventlabels = model.eventlabels;
        sj_res{cond_i} = sj;
    end
    res{subject_i} = sj_res;
end

%% Save the results:
results = [];
condition_columns = ["SOA", "duration", "lock", "task"];
for sub_i = 1:length(res)
    for cond_i = 1:length(res{sub_i})
        T = table();
        % Extract the subject ID:
        T.subject = res{sub_i}{cond_i}.subject;
        % Extract the conditions:
        condition = cellstr(res{sub_i}{cond_i}.conditions{1});
        condition_split = split(condition{1}, '_');
        num_conditions = numel(condition_split);
        for i = 1:num_conditions
            T.(condition_columns{i}) = condition_split(i);
        end
        % Extract the events labels:
        eventlabels = res{sub_i}{cond_i}.estim.(condition{1}).eventlabels;
        % Sort the event labels:
        [eventlabels, I] = sort(eventlabels);
        % Extract the betas:
        betas = res{sub_i}{cond_i}.estim.(condition{1}).ampvals;
        betas = betas(I);
        num_betas = numel(betas);
        for i = 1:num_betas
            T.("beta-" + eventlabels{i}) = betas(i);
        end
        % Extract the latencies:
        latencies = res{sub_i}{cond_i}.estim.(condition{1}).latvals;
        latencies = latencies(I);
        num_latencies = numel(latencies);
        for i = 1:num_latencies
            T.("tau-" + eventlabels{i}) = latencies(i);
        end
        results = [results; T];
    end
end

writetable(results, fullfile(root, sprintf("ses-%s_task-%s_desc-deconvolution_res.csv", session, task)))

