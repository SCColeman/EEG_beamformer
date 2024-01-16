clear all
close all

%%% add fieldtrip path %%%
addpath("C:\Users\ppysc6\OneDrive - The University of Nottingham\Documents\MATLAB\fieldtrip\fieldtrip-20230118")

subject = '11074';

%%% set data paths
data_path = strcat('R:\DRS-PSR\Seb\EEGOPM\data\', subject, '\');
out_path = strcat('R:\DRS-PSR\Seb\EEGOPM\derivatives\', subject, '\forward_model\');
if ~exist(out_path)
    mkdir(out_path)
end

anat_name = strcat(subject, '.nii'); % make sure these are correct
pos_name = strcat(subject, '.pos');

%%% set ft defaults
ft_defaults

%% load data 

%%% load MRI and pos file %%%
mri = ft_read_mri(strcat(data_path, anat_name));
headshape = ft_read_headshape(strcat(data_path, pos_name));
elec = ft_read_sens(strcat(data_path, pos_name), 'senstype','eeg');
elec_index = length(elec.label)-66:length(elec.label);
elec_fields = fields(elec);
for field = 1:length(elec_fields)-2
    field_new = elec.(elec_fields{field});
    elec.(elec_fields{field}) = field_new(elec_index,:);
end

%%% plot headshape and elec
figure
hold on
plot3(headshape.pos(:,1), headshape.pos(:,2), headshape.pos(:,3), '.', ...
    'MarkerSize',4, 'color',[0.5 0.5 0.5])
plot3(elec.chanpos(:,1), elec.chanpos(:,2), elec.chanpos(:,3), 'ro')
text(elec.chanpos(:,1), elec.chanpos(:,2), elec.chanpos(:,3), elec.label,'color','k')
plot3(headshape.fid.pos(:,1), headshape.fid.pos(:,2), headshape.fid.pos(:,3), 'bo')
text(headshape.fid.pos(:,1), headshape.fid.pos(:,2), headshape.fid.pos(:,3),...
    headshape.fid.label,'color','k')

%% put MRI in neuromag coords

cfg=[];
cfg.viewresult='yes';
cfg.method='interactive';
cfg.coordsys='neuromag';
mri_realigned=ft_volumerealign(cfg,mri)

mri_nas = ft_warp_apply(mri_realigned.transform, [mri_realigned.cfg.fiducial.nas]);
mri_lpa = ft_warp_apply(mri_realigned.transform, [mri_realigned.cfg.fiducial.lpa]);
mri_rpa = ft_warp_apply(mri_realigned.transform, [mri_realigned.cfg.fiducial.rpa]);

%% crop MRI

ft_sourceplot([], mri_realigned);
mri_cropped = mri_realigned;
crop = "n";
while crop == "n"
    mri_cropped = mri_realigned;
    cropaxis = input("Which axis needs cropping (inferior-superior axis)? (x,y,z) ", "s");
    cropval = input("Choose crop value");
    if cropaxis == "x"
        mri_cropped.anatomy(1:cropval,:,:) = 0;
    elseif cropaxis == "y"
        mri_cropped.anatomy(:,1:cropval,:) = 0;
    elseif cropaxis == "z"
        mri_cropped.anatomy(:,:,1:cropval) = 0;
    end
    ft_sourceplot([], mri_cropped);
    crop = input("Happy (y/n) ", "s");
end

mri_realigned = mri_cropped;

%% Perform a rough coreg using fiducials to move EEG to MRI space

%%% fiducials from electrode/headshape 
elec_nas = headshape.fid.pos(strcmp(headshape.fid.label,"nas"),:);
elec_lpa = headshape.fid.pos(strcmp(headshape.fid.label,"lpa"),:);
elec_rpa = headshape.fid.pos(strcmp(headshape.fid.label,"rpa"),:);

%%% make template electrode structure
elec_fixed = [];
elec_fixed.elecpos = [
  mri_nas
  mri_lpa
  mri_rpa
  ];
elec_fixed.label = {'nasion', 'left', 'right'};
elec_fixed.unit  = 'mm';

% coregister the moving fids (headshape space) to the fixed fids (MRI space)
cfg = [];
cfg.method   = 'fiducial';
cfg.template = elec_fixed;
cfg.elec     = elec;
cfg.fiducial = {'nasion', 'left', 'right'};
elec_new = ft_electroderealign(cfg);

%%% check partially co-registered electrodes with MRI fids

elec_fixed = ft_convert_units(elec_fixed,'cm');
mri_realigned = ft_convert_units(mri_realigned,'cm');

%%% plot new elec positions and MRI fids
ft_determine_coordsys(mri_realigned,'interactive','no')
hold on
plot3(elec_new.chanpos(:,1), elec_new.chanpos(:,2), elec_new.chanpos(:,3), 'ro')
text(elec_new.chanpos(:,1), elec_new.chanpos(:,2), elec_new.chanpos(:,3), elec_new.label,'color','k')
plot3(elec_fixed.elecpos(:,1), elec_fixed.elecpos(:,2), elec_fixed.elecpos(:,3), 'bo')
text(elec_fixed.elecpos(:,1), elec_fixed.elecpos(:,2), elec_fixed.elecpos(:,3),...
    elec_fixed.label,'color','k')

%% refine coregistration using ICP

%%% apply transform to headshape points
headshape_new = [];
headshape_new.unit = 'cm';
headshape_new.pos = ft_warp_apply(elec_new.homogeneous, headshape.pos(1:20:end,:));

%%% plot 
ft_determine_coordsys(mri_realigned,'interactive','no')
hold on
plot3(headshape_new.pos(:,1), headshape_new.pos(:,2), headshape_new.pos(:,3), 'r.')

%%% refine
cfg=[];
cfg.method='headshape';
cfg.elec = elec_new;
cfg.headshape.headshape=headshape_new;
cfg.headshape.icp='yes';
cfg.coordsys='neuromag';
mri_refined =ft_volumerealign(cfg, mri_realigned);

%%% plot electrodes with refined MRI
ft_determine_coordsys(mri_refined,'interactive','no')
hold on
plot3(elec_new.chanpos(:,1), elec_new.chanpos(:,2), elec_new.chanpos(:,3), 'ro')
text(elec_new.chanpos(:,1), elec_new.chanpos(:,2), elec_new.chanpos(:,3), elec_new.label,'color','k')
plot3(elec_fixed.elecpos(:,1), elec_fixed.elecpos(:,2), elec_fixed.elecpos(:,3), 'bo')
text(elec_fixed.elecpos(:,1), elec_fixed.elecpos(:,2), elec_fixed.elecpos(:,3),...
    elec_fixed.label,'color','k')

%% make layer segmentations

cfg=[];
cfg.output = {'brain', 'skull', 'scalp'};
%cfg.skullsmooth = 20;   %%% Set these if headmodel has nans in it,
%cfg.brainsmooth = 15;   %%% it's likely that there's gaps in one of the 
%cfg.scalpsmooth = 20;   %%% three layers. 10 is a good starting point.
mri_segm=ft_volumesegment(cfg,mri_refined)

%%% check segmentation
mri_segm_indexed = ft_checkdata(mri_segm, 'segmentationstyle', 'indexed')
mri_segm_indexed.anatomy = mri_refined.anatomy;

cfg = [];
cfg.method = 'ortho';
cfg.anaparameter = 'anatomy';
cfg.funparameter = 'tissue';
cfg.funcolormap = [
  0 0 0
  1 0 0
  0 1 0
  0 0 1
  ];
ft_sourceplot(cfg, mri_segm_indexed)

%% make layer meshes

cfg = [];
cfg.tissue      = {'brain', 'skull' 'scalp'};
cfg.numvertices = [3000 2000 1000];
mesh = ft_prepare_mesh(cfg, mri_segm);

%%% plot meshes
figure
ft_plot_mesh(mesh(1), 'facecolor','r', 'facealpha', 1.0, 'edgecolor', 'k', 'edgealpha', 1);
hold on
ft_plot_mesh(mesh(2), 'facecolor','g', 'facealpha', 0.4, 'edgecolor', 'k', 'edgealpha', 0.1);
hold on
ft_plot_mesh(mesh(3), 'facecolor','b', 'facealpha', 0.4, 'edgecolor', 'k', 'edgealpha', 0.1);
hold on
ft_plot_sens(elec_new, 'label', 'label', 'elecshape', 'circle', 'facecolor', [0.8 0.3 0.3])

%% make volume conduction model from meshes

% Create a volume conduction model
cfg        = [];
cfg.method = 'bemcp'; % You can also specify 'openmeeg', 'bemcp', or another method
headmodel  = ft_prepare_headmodel(cfg, mesh);

%% calculate leadfield matrix

cfg=[];
cfg.elec=elec_new;
cfg.headmodel=headmodel;
cfg.grid.resolution=0.4;
cfg.grid.unit='cm';
cfg.normalize='yes';
grid=ft_prepare_leadfield(cfg);

%%% plot
figure
ft_plot_sens(elec_new, 'label', 'label', 'elecshape', 'circle', 'facecolor', [0.8 0.3 0.3])
ft_plot_headmodel(headmodel, 'edgecolor', 'none'); alpha 0.4;
ft_plot_mesh(grid.pos(grid.inside,:));

%% downsample MRI to match functional %%%

mri_mm = ft_convert_units(mri_refined, 'mm');
mri_highres = ft_volumereslice([],mri_mm);
grid_mm = ft_convert_units(grid, 'mm');
cfg = [];
cfg.xrange = [min(grid_mm.pos(:,1)) max(grid_mm.pos(:,1))];
cfg.yrange = [min(grid_mm.pos(:,2)) max(grid_mm.pos(:,2))];
cfg.zrange = [min(grid_mm.pos(:,3)) max(grid_mm.pos(:,3))];
cfg.resolution = 4;
mri_lowres = ft_volumereslice(cfg,mri_mm);
mri_lowres.anatomy(grid_mm.inside==0) = 0;   %%% strip skull/csf
ft_sourceplot([],mri_lowres)

%% save 

save(strcat(out_path, subject, '_headmodel.mat'), 'headmodel')
save(strcat(out_path, subject, '_LF_grid.mat'), 'grid')
save(strcat(out_path, subject, '_mri_reg.mat'), 'mri_refined')
save(strcat(out_path, subject, '_elec_reg.mat'), 'elec_new')

% downsampled anat
anat_fname = strcat(out_path, subject, '_anatomy_lowres');
ft_write_mri(anat_fname, mri_lowres.anatomy, 'transform', mri_lowres.transform, 'dataformat', 'nifti')

% highres anat
anat_fname = strcat(out_path, subject, '_anatomy_highres');
ft_write_mri(anat_fname, mri_highres.anatomy, 'transform', mri_highres.transform, 'dataformat', 'nifti')


