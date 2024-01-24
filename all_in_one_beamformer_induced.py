# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:54:55 2024

@author: ppysc6
"""

import os
import os.path as op

import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, inspect_dataset
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

events = mne.events_from_annotations(data)[0]
mne.viz.plot_events(events)

#%% make appropriate montage (replace later with digitisation)

montage = mne.channels.make_standard_montage("easycap-M1")
montage.plot()
data.set_montage(montage, on_missing="ignore")

#%% basic preprocessing

# remove ECG (comment out if using this)
data.drop_channels("ECG")

# set reference 
data.set_eeg_reference('average')

orig_freq = 1000
sfreq = 250
data_ds = data.copy().resample(sfreq=sfreq)
events[:,0] = np.round(events[:,0] * (sfreq/orig_freq))
data_filt = data_ds.copy().filter(l_freq=1, h_freq=48)
data_filt.plot_psd(fmax=48, picks='eeg').show()

#%%  ICA

data_ica = data_filt.copy()
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(data_ica.pick_types(eeg=True))
ica.plot_components()
ica.plot_sources(data_ica)

#%% remove ICA

ica.exclude = [0, 7]   # CHANGE THESE TO ECG/CARDIAC COMPONENTS
ica.apply(data_ica)

#%% annotate muscle artifacts

data_annot = data_ica.copy()
threshold_muscle = 20  # z-score
annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    data_annot,
    threshold=threshold_muscle,   # zscore
    ch_type = "eeg",
    min_length_good=2,
    filter_freq=(50, 80)
)

fig, ax = plt.subplots()
ax.plot(data_annot.times, scores_muscle)
ax.axhline(y=threshold_muscle, color="r")
ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")

data_annot.set_annotations(annot_muscle)

#%% set proper montage using .pos file

elec_names = np.array(["Fp1","Fpz","Fp2","AF7","AF3","AF4","AF8","F7","F5","F3","F1",
    "Fz","F2","F4","F6","F8","FT7","FC5","FC3","FC1","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","Cz","C2","C4","C6","T8","TP9","TP7","CP5","CP3",
    "CP1","CPz","CP2","CP4","CP6","TP8","TP10","P7","P5","P3","P1","Pz","P2",
    "P4","P6","P8","PO9","PO7","PO3","POz","PO4","PO8","PO10","O1","Oz","O2"])

# load in pos file to pandas dataframe
df = pd.read_table(op.join(data_path, subject, subject + '.pos'), names=['point','x','y','z'])
df = df.drop(df.index[0])

# extract electrodes and fids from dataframe
fid_positions = df[['x', 'y', 'z']].values[len(df)-3:len(df)] / 100
fid_labels = df['point'].values[len(df)-3:len(df)]
n_electrodes = 63
elec_positions = df[['x', 'y', 'z']].values[len(df)-n_electrodes-3:len(df)-3] / 100
elec_dict = dict(zip(elec_names,elec_positions))
head_positions = df[['x', 'y', 'z']].values[0:len(df)-n_electrodes-4] / 100
head_positions_ds = head_positions[0::100,:]  # heavily downsampled

# create head digitisation
digitisation = mne.channels.make_dig_montage(ch_pos=elec_dict, 
                         nasion=np.squeeze(fid_positions[fid_labels=='nasion ',:]),
                         lpa=np.squeeze(fid_positions[fid_labels=='left ',:]),
                         rpa=np.squeeze(fid_positions[fid_labels=='right ',:]),
                         hsp=head_positions_ds)

data_preproc = data_annot.copy()
data_preproc.set_montage(digitisation, on_missing="ignore")
data_preproc.save(op.join(preproc_out, "data_preproc_" + task + ".fif"),
                   overwrite=True)

#%% epochs

#data_preproc = mne.io.Raw(op.join(preproc_out, "data_preproc_" + task + ".fif"))

event_id = 65     # trigger of interest
fband = [8, 13]
tmin, tmax = 0, 40
epochs = mne.Epochs(
    data_preproc,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True,
    reject_by_annotation=True)

epochs_filt = epochs.filter(fband[0], fband[1]).copy()

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
info = data_preproc.info

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

#%% forward solution

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

#%% spatial filter

filters = mne.beamformer.make_lcmv(
    info,
    fwd,
    all_cov,
    reg=0.05,
    noise_cov=None,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None)

#%% apply filter to covariance for static sourcemap
 
stc_active = mne.beamformer.apply_lcmv_cov(active_cov, filters)
stc_base = mne.beamformer.apply_lcmv_cov(control_cov, filters)
 
stc_change = (stc_active - stc_base) / (stc_active + stc_base)
 
#%% visualise static sourcemap
 
stc_change.plot(src=src, subject=fs_subject,
                subjects_dir=subjects_dir, surface="pial", hemi="both")

#%% apply beamformer to filtered raw data for timecourse extraction
 
stc_raw = mne.beamformer.apply_lcmv_raw(data_preproc.set_eeg_reference('average',
                                                     projection=True), filters)
 
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
power[0].plot(picks="all", baseline=(25, 35))

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