%%% Script for plotting voxel TFS and timecourses using beamformer     %%%
%%% outputs from lcmv_beamforming_2.m                                  %%%
%%% Author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk              %%%

clear all
close all

%%% add fieldtrip path %%%
addpath("C:\Users\ppysc6\OneDrive - The University of Nottingham\Documents\MATLAB\fieldtrip\fieldtrip-20230118")

subject = '11074';

%%% set data paths
data_path = strcat('R:\DRS-PSR\Seb\EEGOPM\data\', subject, '\');
source_path = strcat('R:\DRS-PSR\Seb\EEGOPM\derivatives\', subject, '\source_model\');

%%% task variables
band = 'beta';
band_lims = [13 30]; % for plotting timecourse
task_identifier = 'FC_still';
contrast_identifier = 'circles';
baseline_window = [0 0.3]; % for plotting TFS

%%% set ft defaults
ft_defaults

%% load source results

%%% unfiltered data
data_fname = strcat(source_path, subject, '_data_preproc_', task_identifier, ...
     '_', contrast_identifier, '_unfiltered');
load(data_fname)

%%% source map
T_fname = strcat(source_path, subject, '_pseudoT_', task_identifier, '_', ...
    contrast_identifier, '_', char(band), '.mat');
load(T_fname)

%%% weights
w_fname = strcat(source_path, subject, '_weights_', task_identifier, '_', ...
    contrast_identifier, '_', char(band), '.mat');
load(w_fname)

%% plot sourcemap and choose/detect voxel of interest

baseline_ind = data_unfilt.time{1} > baseline_window(1) & ...
    data_unfilt.time{1} <= baseline_window(2);

cfg=[];
cfg.method='ortho';
cfg.funparameter='pow';
ft_sourceplot(cfg,T_mm);

%%% change "voxel" to number in voxel field in sourcemap plot if you want
%%% to plot specific voxel
voxel = find(abs(T_mm.avg.pow)==max(abs(T_mm.avg.pow))); % this plots abs peak
%voxel = find(T.avg.pow == max(T.avg.pow));  % only positive peak
%voxel = find(T.avg.pow == min(T.avg.pow));  % only negative peak
%voxel = 23510 %14809;
voxel_W = w{voxel};

timecourse = [];
for trial = 1:length(data_unfilt.trial)
    timecourse(trial,:) = voxel_W*data_unfilt.trial{trial};
end

%%% TFS
freqs_lower = [2 4 6 8 10 14 18 22 26 30 35 40 45];
freqs_higher = [4 6 8 10 14 18 22 26 30 35 40 45 50];
freqs_centre = (freqs_lower + freqs_higher) / 2;
TFS = zeros(length(freqs_centre), length(data_unfilt.time{1}));
for freq = 1:length(freqs_centre)
    timecourse_filt = bandpass(timecourse', [freqs_lower(freq) freqs_higher(freq)], data_unfilt.fsample)';
    timecourse_hilb = abs(hilbert(timecourse_filt'))';
    timecourse_avg = mean(timecourse_hilb, 1);
    timecourse_relative = (timecourse_avg - mean(timecourse_avg(baseline_ind)))./mean(timecourse_avg(baseline_ind));
    TFS(freq, :) = timecourse_relative;
end

figure('color','w')
pcolor(data_unfilt.time{1}, freqs_centre, TFS);
shading interp;
pbaspect([1 1 1])
set(gca, 'FontSize', 14)
xlabel('Time (s)','FontSize',16)
ylabel('Frequency (Hz)','FontSize',16)
title(strcat("Peak ", band, " VE") ,'FontSize',16)

%% plot timecourse

freq = band_lims;
timecourse_filt = bandpass(timecourse', freq, data_unfilt.fsample)';
timecourse_hilb = abs(hilbert(timecourse_filt'))';
timecourse_avg = mean(timecourse_hilb, 1);

figure('color','w')
plot(data_unfilt.time{1}, timecourse_avg, 'LineWidth', 2)
xlabel('Time (s)')
ylabel('Power (A.U)')
set(gca, 'FontSize', 16)
