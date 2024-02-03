# -*- coding: utf-8 -*-
"""
A script to perform pre-processing, forward modelling and beamforming on
BrainVision EEG files. The script requires a FreeSurfer reconstruction, or
use FSaverage (as it is set here) for testing purposes.

For actual study usage, it is recommended to split this script into several
parts, e.g. pre-processing, forward model, source reconstruction.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os
import os.path as op
import numpy as np
import mne
from matplotlib import pyplot as plt
import mne.datasets
import pandas as pd

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up path

data_path = r'R:\DRS-PSR\Seb\EEGOPM\data'
deriv_path = r'R:\DRS-PSR\Seb\EEGOPM\derivatives'

# scanning session info
subject = '11766'
task = 'nback'  # name of the task
fname = subject + "_" + task

# create paths
preproc_out = op.join(deriv_path, subject, "preprocessing")
if not op.exists(preproc_out):
    os.makedirs(preproc_out)
forward_out = op.join(deriv_path, subject, "forward_model")
if not op.exists(forward_out):
    os.makedirs(forward_out)
source_out = op.join(deriv_path, subject, "source_model")
if not op.exists(source_out):
    os.makedirs(source_out)

#%% load data and events

data = mne.io.read_raw_brainvision(op.join(data_path,subject,fname + '.vhdr'),
                                   preload=True)
data.info
data_raw = data.copy()  # these lines are added to maintain a copy of the data
                        # at each step - useful for debugging.
                        
### events from marker file
events = mne.events_from_annotations(data)[0]
mne.viz.plot_events(events)

#%% make appropriate montage (replace later with digitisation)

montage = mne.channels.make_standard_montage("easycap-M1")
montage.plot()
data.set_montage(montage, on_missing="ignore")
data_montage = data.copy()

#%% remove ECG

data.drop_channels("ECG")

#%% downsample

orig_freq = data.info["sfreq"]
sfreq = 500
data.resample(sfreq=sfreq)
data_ds = data.copy()

### manually downsample events
events[:,0] = np.round(events[:,0] * (sfreq/orig_freq))

#%% broadband filter

data.filter(l_freq=1, h_freq=150)
data_broadband = data.copy()

#%% plot data and left click any bad channels

data.plot()

#%% set average reference

data.set_eeg_reference('average', projection=True)
data_ref = data.copy()

#%% plot PSD

data.plot_psd(fmax=48, picks='eeg').show()

#%% fit ICA, CLICK ON BAD COMPONENT TIMECOURSES TO MARK AS BAD

ica = mne.preprocessing.ICA(n_components=20)
ica.fit(data.pick_types(eeg=True))
ica.plot_components()
ica.plot_sources(data)

#%% apply ICA 

ica.apply(data)
data_ica = data.copy()

#%% annotate muscle artifacts

threshold_muscle = 20  # z-score
annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    data,
    threshold=threshold_muscle,   # zscore
    ch_type = "eeg",
    min_length_good=2,
    filter_freq=(100, 130)
)

fig, ax = plt.subplots()
ax.plot(data.times, scores_muscle)
ax.axhline(y=threshold_muscle, color="r")
ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")

#%% set bad muscle annotations IF HAPPY WITH THRESHOLDING ABOVE

data.set_annotations(annot_muscle)
data_muscle = data.copy()

#%% set proper montage using .pos file, CREATED FROM AN EINSCAN FILE, MODIFY ACCORDINGLY

elec_names = data_ds.ch_names

# load in pos file to pandas dataframe
df = pd.read_table(op.join(data_path, subject, subject + '.pos'), 
                   names=['point','x','y','z'], delim_whitespace=True)
pos = df.drop(df.index[0]).to_numpy()

# separate pos into fiducials, electrodes and headshape
# 3 fiducial points at the end
# 1 points for each channel (63 channels)
# the rest are headshape points

pos_fids = pos[-3:,1:] / 100  # change units to m for MNE
pos_elec = pos[-3-len(elec_names):-3,1:] / 100

pos_head = pos[0::100,1:] / 100   # downsample Einscan by 100 for speed

# divide pos by 100 
elec_dict = dict(zip(elec_names,pos_elec))

nas = pos_fids[0,:].astype(float)
lpa = pos_fids[1,:].astype(float)
rpa = pos_fids[2,:].astype(float)
hsp = pos_head.astype(float)

# create head digitisation
digitisation = mne.channels.make_dig_montage(ch_pos=elec_dict, 
                         nasion=nas,
                         lpa=lpa,
                         rpa=rpa,
                         hsp=hsp)

data.set_montage(digitisation, on_missing="ignore")
data_dig = data.copy()

#%% general preprocessing is now done, save out preproc file

data.save(op.join(preproc_out, "data_preproc_" + task + ".fif"),
                   overwrite=True)
info = data.info # need this for later sections

#%% epoch data based on trigger

#data = mne.io.Raw(op.join(preproc_out, "data_preproc_" + task + ".fif"))

event_id = 65     # trigger of interest
tmin, tmax = 0, 40
epochs = mne.Epochs(
    data,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True,
    reject_by_annotation=True)

#%% prefilter epochs before calculating covariance

fband = [8, 13]
epochs.filter(fband[0], fband[1])
epochs_filt = epochs.copy()

#%% compute covariance

act_min, act_max = 5, 15
con_min, con_max = 25, 35

active_cov = mne.compute_covariance(epochs_filt, tmin=act_min, tmax=act_max, method="shrunk")
control_cov= mne.compute_covariance(epochs_filt, tmin=con_min, tmax=con_max, method="shrunk")
all_cov = mne.compute_covariance(epochs_filt, method="shrunk")
all_cov.plot(epochs_filt.info)

#%% Get FS Average for forward model

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
fs_subject = 'fsaverage'

plot_bem_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200])

mne.viz.plot_bem(**plot_bem_kwargs)

#%% coregistration

mne.gui.coregistration(subjects_dir=subjects_dir, subject=fs_subject, scale_by_distance=False)

#%% visualise coreg

trans = op.join(forward_out, subject + "_" + task + "-trans.fif")

mne.viz.plot_alignment(
    info,
    trans,
    subject=fs_subject,
    dig=True,
    subjects_dir=subjects_dir,
    surfaces="head-dense")

#%% compute source space

surf_file = op.join(subjects_dir, fs_subject, "bem", "inner_skull.surf")

# can change oct5 to other surface source space
src = mne.setup_source_space(
    fs_subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir)
src.plot(subjects_dir=subjects_dir)

#%% Calculate lead field using 3 layer model

conductivity = (0.3, 0.006, 0.3)
model = mne.make_bem_model(
    subject=fs_subject, ico=4,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(
    info,
    trans=trans,
    src=src,
    bem=bem,
    meg=False,
    eeg=True,
    mindist=5,
    )
print(fwd)

#%% calculate beamformer weights (spatial filter)

filters = mne.beamformer.make_lcmv(
    info,
    fwd,
    all_cov,
    reg=0.05,  # 5% regularisation
    noise_cov=None,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None)

#%% apply beamformer wweights to covariance for static sourcemap
 
stc_active = mne.beamformer.apply_lcmv_cov(active_cov, filters)
stc_base = mne.beamformer.apply_lcmv_cov(control_cov, filters)
 
stc_change = (stc_active - stc_base) / (stc_active + stc_base)
 
#%% visualise static sourcemap
 
stc_change.plot(src=src, subject=fs_subject,
                subjects_dir=subjects_dir, surface="pial", hemi="both")

#%% apply beamformer to filtered raw data for timecourse extraction
 
stc_raw = mne.beamformer.apply_lcmv_raw(data, filters)
 
#%% extract absolute max voxel TFS/timecourse
 
peak = stc_change.get_peak(mode="abs", vert_as_index=True)[0]
 
stc_peak = stc_raw.data[peak]

# make fake raw object from source time course
 
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=info["sfreq"],
                              ch_types=ch_types)
source_raw = mne.io.RawArray([stc_peak], source_info)
 
source_epochs = mne.Epochs(
    source_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True)
 
 
# TFR
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all"
                                           )
power[0].plot(picks="all", baseline=(30, 38))

### timecourse

source_epochs_filt = source_epochs.filter(fband[0], fband[1], picks="all").copy()
source_epochs_filt.apply_hilbert(envelope=True, picks="all")
stc_epochs_filt = source_epochs_filt.average(picks="all")
stc_epochs_filt.plot()

#%% extract maximum from within a parcel

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

# get induced peak within label
label = 15
stc_inlabel = stc_change.in_label(labels[label])
label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]

# extract timecourse of peak
stc_label_all = mne.extract_label_time_course(stc_raw, labels[label], src, mode=None)
stc_label_peak = stc_label_all[0][label_peak,:]

# make fake raw object from source time course
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=info["sfreq"],
                              ch_types=ch_types)
source_label_raw = mne.io.RawArray([stc_label_peak], source_info)
 
source_label_epochs = mne.Epochs(
    source_label_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True)
 
 
# TFR
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_label_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all")
power[0].plot(picks="all", baseline=(25, 35))

### timecourse
source_label_filt = source_label_epochs.filter(fband[0], fband[1], picks="all").copy()
source_label_filt.apply_hilbert(envelope=True, picks="all")
stc_label_filt = source_label_filt.average(picks="all")
stc_label_filt.plot()
