clear all
close all

%%% add fieldtrip path %%%
addpath("C:\Users\ppysc6\OneDrive - The University of Nottingham\Documents\MATLAB\fieldtrip\fieldtrip-20230118")

subject = '11074';

%%% set data paths
data_path = strcat('R:\DRS-PSR\Seb\EEGOPM\data\', subject, '\');
forward_path = strcat('R:\DRS-PSR\Seb\EEGOPM\derivatives\', subject, '\forward_model\');
out_path = strcat('R:\DRS-PSR\Seb\EEGOPM\derivatives\', subject, '\source_model\');
if ~exist(out_path)
    mkdir(out_path)
end

%%% paradigm details (best to save a copy of the script with these changed
%%% for each of the contrasts you're going to be analysing)
task_identifier = 'FC_still';
contrast_identifier = 'circles';
trigger_of_interest = "S 96";  % look in brainvision .vmrk file "S 65" - 2back start, "S 96" - circles
prestim = 0 - 0.3; % time BEFORE trigger for data segmentation
poststim = 2 + 0.3; % time AFTER trigger for data segmentation
active_win = [0+0.3 0.5+0.3]; % for T-stat contrast, in seconds
control_win = [1.5+0.3 2+0.3]; % for T-stat contrast, in seconds
file_name = strcat(subject, '_faces_circles_still');   % basename for eeg files
hdr_name = strcat(file_name, '.vhdr');
dat_name = strcat(file_name, '.eeg');
mrk_name = strcat(file_name, '.vmrk');

%%% set ft defaults
ft_defaults

%%% load headmodel outputs from forward_model_1.m script
load(strcat(forward_path, subject, '_headmodel.mat'))
load(strcat(forward_path, subject, '_LF_grid.mat'))
load(strcat(forward_path, subject, '_mri_reg.mat'))
load(strcat(forward_path, subject, '_elec_reg.mat'))

%% pre-processing steps common to all contrasts/frequency bands

perform_preprocess = input("Perform initial preprocessing (common to all contrasts/bands)? (y/n) ", "s");

if perform_preprocess == 'y'
    
    %%% load data and do basic pre-processing
    cd(data_path)
    cfg = [];
    cfg.headerfile   = hdr_name;
    cfg.datafile     = dat_name;
    cfg.channel = {'all', '-ecg'};
    cfg.bpfilter = 'yes';
    cfg.bpfreq = [1 150];
    data = ft_preprocessing(cfg);

    %%% remove bad channels visually
    cfg = ft_databrowser([], data);
    drawnow;
    waitfor(cfg)
    [indx, tf] = listdlg('ListString', data.label, 'PromptString', "Select Bad Channels");
    keepchans = data.label;
    keepchans(indx) = [];

    %%% remove
    cfg = [];
    cfg.channel = keepchans;
    data = ft_selectdata(cfg, data);

    %%% Re-reference EEG to average of all (apart from bad) channels
    cfg = [];
    cfg.reref = 'yes';
    cfg.channel = {'all'}; 
    cfg.refmethod = 'avg';
    cfg.refchannel = {'all'};
    data = ft_preprocessing(cfg, data);

    %%% ICA to remove eyeblinks and any cardiac 
    data_rank= rank(data.trial{1}*data.trial{1}');
    cfg        = [];  
    cfg.method = 'runica';  
    cfg.numcomponent = data_rank;
    data_comps = ft_componentanalysis(cfg,data);   

    cfg = [];
    cfg.channel = [1:10]; 
    cfg.continuous='no';
    cfg.viewmode = 'component'; 
    cfg.layout = 'easycapM11.mat';
    ft_databrowser(cfg, data_comps);

    %%% reject bad components %%%
    cfg = [];  
    cfg.component = input("Type array of cardiac/blink components ");  %%% INPUT BLINK/CARDIAC COMPONENT
    data = ft_rejectcomponent(cfg, data_comps, data);

    %%% save data after common pre-processing %%%
    data_fname = strcat(out_path, subject, '_data_common_preproc_', task_identifier);
    save(data_fname, 'data')
else
    data_fname = strcat(out_path, subject, '_data_common_preproc_', task_identifier);
    load(data_fname)
end

%% find bad trials

%%% segment into trials based on trigger
cd(data_path)
cfg              = [];
cfg.headerfile   = hdr_name;
cfg.datafile     = dat_name;
cfg.trialfun = 'trialfun_returntrig';    % my custom function
cfg.trialdef.eventtype = 'Stimulus';
cfg.trialdef.eventvalue = {trigger_of_interest};   % trial start trigger
cfg.trialdef.prestim = prestim;   % time window
cfg.trialdef.poststim = poststim;
cfg_trialdef = ft_definetrial(cfg);

%%% chop data into segments at this stage JUST FOR BAD TRIAL SELECTION
data_seg = ft_redefinetrial(cfg_trialdef, data);

ft_databrowser([], data_seg);
badtrials = input("Type Array of Bad Trials ");

%%% get array of good trials (to keep) for later
goodtrials = 1:length(data_seg.trial);
for bad = 1:length(badtrials)
    goodtrials(goodtrials == badtrials(bad)) = [];
end

%% finish pre-processing broadband data and save out

data_unfilt = data;

%%% chop data into segments based on earlier trial definition
data_unfilt = ft_redefinetrial(cfg_trialdef, data_unfilt);

%%% get rid of bad trials
cfg = [];
cfg.trials = goodtrials;
data_unfilt = ft_selectdata(cfg, data_unfilt);

%%% downsample to 500 Hz
cfg = [];
cfg.resamplefs = 500;
data_unfilt = ft_resampledata(cfg, data_unfilt);

data_fname = strcat(out_path, subject, '_data_preproc_', task_identifier, '_', ...
    contrast_identifier, '_unfiltered');
save(data_fname, 'data_unfilt')

%% Beamform using different frequency bands, plot and save

bands = ["theta", "alpha", "beta", "gamma"];
bp_low = [4 8 13 30];
bp_high = [8 13 30 80];

for band = 1:length(bands)
    
    %%% bandpass filter continuous data
    cfg = [];
    cfg.bpfilter = 'yes';
    cfg.bpfreq = [bp_low(band) bp_high(band)];
    data_filt = ft_preprocessing(cfg, data);

    %%% chop data into segments based on earlier trial definition
    data_filt = ft_redefinetrial(cfg_trialdef, data_filt);
    
    %%% get rid of bad trials
    cfg = [];
    cfg.trials = goodtrials;
    data_filt = ft_selectdata(cfg, data_filt);
    
    %%% downsample to 500 Hz
    cfg = [];
    cfg.resamplefs = 500;
    data_filt = ft_resampledata(cfg, data_filt);
    
    %%% Make beamformer weights using covariance from all data %%%
    
    act_win = active_win;  % in seconds
    con_win = control_win;
    
    %%% calculate covariance of active window
    cfg=[];
    cfg.latency = [act_win(1)+(1/data_filt.fsample) act_win(2)];
    data_act=ft_selectdata(cfg,data_filt);
    
    %%% calculate covariance of control window
    cfg=[];
    cfg.latency = [con_win(1)+(1/data_filt.fsample) con_win(2)];
    data_con=ft_selectdata(cfg,data_filt);
    
    %%% timelock analysis %%%    
    cfg = [];
    cfg.covariance = 'yes';
    timelock_all = ft_timelockanalysis(cfg,data_filt);
    timelock_act = ft_timelockanalysis(cfg,data_act);
    timelock_con = ft_timelockanalysis(cfg,data_con);
    
    %%% lcmv %%%    
    cfg = [];
    cfg.method = 'lcmv';
    cfg.sourcemodel = grid;
    cfg.headmodel = headmodel;
    cfg.elec = elec_new;
    cfg.lcmv.lambda       = ['5' '%']; %regularisation, for pseudo T
    cfg.lcmv.keepfilter   = 'yes'; %keeps the beamformer weights
    cfg.lcmv.fixedori     ='yes'; %does scalar beamformer... if 'no vector beamformer. leadfield for the optimal dipole orientation
    cfg.lcmv.projectmom   = 'yes'; %project the dipole moment timecourse on the direction of maximal power
    cfg.lcmv.keepmom      = 'no'; %dont store the VE for every grid point as calculate the beamformer
    cfg.lcmv.projectnoise = 'yes'; %provide the noise estimate %added 16/11
    source = ft_sourceanalysis(cfg, timelock_all);
    
    w = source.avg.filter;
    
    source_act = ft_sourceanalysis(cfg, timelock_act);
    source_con = ft_sourceanalysis(cfg, timelock_con);
    
    %%% pseudo T %%%
    
    T = source_act;
    T.avg.pow = (source_act.avg.pow - source_con.avg.pow) ./ (source_act.avg.noise + source_con.avg.noise);
    T_mm = ft_convert_units(T, 'mm');
    
    %%% fancy highres stuff, purely visual %%%
    
    mri_mm = ft_convert_units(mri_refined, 'mm');
    mri_highres = ft_volumereslice([], mri_mm);
    
    %%% interpolate sourcemap on to anatomical, for visual purposes only
    cfg=[];
    cfg.parameter='avg.pow';
    T_highres=ft_sourceinterpolate(cfg, T_mm, mri_highres);
    T_highres.pow(isnan(T_highres.pow)) = 0;
    
    %%% plot sourcemap and peak voxel %%%
    
    %%% plot sourcemap %%%
    cfg=[];
    cfg.method='ortho';
    cfg.funparameter='pow';
    ft_sourceplot(cfg,T_mm);
    
    %%% plot peak voxel over whole brain
    voxel = find(abs(T.avg.pow)==max(abs(T.avg.pow))); % this plots abs peak
    voxel_W = w{voxel};
    
    timecourse = [];
    for trial = 1:length(data_filt.trial)
        timecourse(trial,:) = voxel_W*data_filt.trial{trial};
    end
    
    %%% frequency filter
    timecourse_filt = timecourse;
    timecourse_hilb = abs(hilbert(timecourse_filt'))';
    timecourse_avg = mean(timecourse_hilb, 1);
    
    figure('color','w')
    plot(data_filt.time{1}, timecourse_avg, 'LineWidth', 2)
    xlabel('Time (s)')
    ylabel('Power (A.U)')
    set(gca, 'FontSize', 16)
    
    %%% save out %%%

    %%% lowres func
    T_fname = strcat(out_path, subject, '_pseudoT_', task_identifier, '_', ...
        contrast_identifier, '_', char(bands(band)));
    cfg = [];
    cfg.filename = T_fname;
    cfg.filetype = 'nifti';
    cfg.parameter = 'avg.pow';
    ft_sourcewrite(cfg, T_mm)
    save(strcat(T_fname, '.mat'), 'T_mm')
    
    %%% highres func %%%
    T_fname = strcat(out_path, subject, '_pseudoT_highres_', task_identifier, '_', ...
        contrast_identifier, '_', char(bands(band)));
    cfg = [];
    cfg.filename = T_fname;
    cfg.filetype = 'nifti';
    cfg.parameter = 'pow';
    ft_sourcewrite(cfg, T_highres)
    
    %%% weights
    weights_fname = strcat(out_path, subject, '_weights_', task_identifier, ...
        '_', contrast_identifier, '_', char(bands(band)));
    save(weights_fname, 'w')

end
