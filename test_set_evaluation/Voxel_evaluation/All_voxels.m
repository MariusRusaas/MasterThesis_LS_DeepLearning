%% Calculate the 

clear

orig_dir = 'E:\Master_Project_Marius\Processed_HCP_data\Testing_data\Original_HCP';
LS_dir = 'E:\Master_Project_Marius\Processed_HCP_data\Testing_data\LS_imgs';
DL3D_dir = 'E:\Master_Project_Marius\Processed_HCP_data\Testing_data\Unet_recon_3D\RayTuned\raw_model_output';


filelist_LS = dir(LS_dir);
filelist_LS = filelist_LS(3:end);
filelist_orig = dir(orig_dir);
filelist_orig = filelist_orig(3:end);
filelist_DL3d = dir(DL3D_dir);
filelist_DL3d = filelist_DL3d(3:end);

test_metrics = zeros(length(filelist_DL3d), 4);

test_maps = [];
dl_corr_tot = [];
ls_corr_tot = [];

for sub = 1:length(filelist_DL3d)
    fprintf('================================================================================\n')
    fprintf(string(datetime) + '  --  Loading sub %d ...    \n', sub);
    tic;
    HCP = niftiread([filelist_orig(sub).folder, '\', filelist_orig(sub).name]);
    
    LS = niftiread([filelist_LS(sub).folder, '\', filelist_LS(sub).name]);
    
    DL3d_recon = niftiread([filelist_DL3d(sub).folder, '\', filelist_DL3d(sub).name]);
    fprintf(string(datetime) + '  --  Finished \n\n');

    counts.countDL = 0;
    counts.countLS = 0;
    counts.tot = 0;
    dims = size(HCP);
    ls_hcp = zeros(dims(1:3));
    dl_ls = zeros(dims(1:3));
    dl_hcp = zeros(dims(1:3));
    fprintf(string(datetime) + '  --  Calculating coeffmaps ...    \n');
    prog = 0;
    fprintf(1,'Computation Progress: %3d%%\n',prog);
    for slice = 1:dims(3)
        for row = 1:dims(1)
            for col = 1:dims(2)
                if mean(squeeze(HCP(row,col,slice,:))) > 0.001
                    counts.tot = counts.tot+1;
                    HCP_vox = squeeze(HCP(row, col, slice, :)); 
                    LS_vox = squeeze(LS(row, col, slice, :)); 
                    DL_vox = squeeze(DL3d_recon(row, col, slice, :)); 
% =========================================================================
% Maximum frequency in frequency spectrum of a timeseries with one data
% point per second for 900 hundred seconds is 0.5 Hz (Nyquist sampling)
% 
    
                    ft_HCP = fft(HCP_vox); ft_LS = fft(LS_vox); ft_DL = fft(DL_vox);
                    ft_HCP(1:6) = 0; ft_LS(1:6) = 0; ft_DL(1:6) = 0;
                    ft_HCP(894:900) = 0; ft_LS(894:900) = 0; ft_DL(894:900) = 0;
                    ft_HCP(83:818) = 0; ft_LS(83:818) = 0; ft_DL(83:818) = 0;
                    HCP_voxn = ifft(ft_HCP); LS_voxn = ifft(ft_LS); DL_voxn = ifft(ft_DL);
                    HCP_voxn = real(HCP_voxn) + imag(HCP_voxn);LS_voxn = real(LS_voxn) + imag(LS_voxn); DL_voxn = real(DL_voxn) + imag(DL_voxn);

                    LS_corr = corrcoef(LS_voxn, HCP_voxn);
                    DL_corrH = corrcoef(DL_voxn, HCP_voxn);
                    DL_corr = corrcoef(DL_voxn, LS_voxn);
                    if DL_corrH(1, 2) >= LS_corr(1, 2), counts.countDL = counts.countDL + 1;
                    else, counts.countLS = counts.countLS + 1;
                    end
                    ls_hcp(row, col, slice) = LS_corr(1, 2);
                    dl_hcp(row, col, slice) = DL_corrH(1, 2);
                    dl_ls(row, col, slice) = DL_corr(1, 2);
                    test_metrics(sub, :) = [sub, counts.countDL, counts.countLS, counts.tot];
                end
            end
        end
        prog = ( 100*(slice/dims(3)) );
	    fprintf(1,'\b\b\b\b%3.0f%%',prog);
    end
    diffmap = dl_hcp - ls_hcp;
    dl_corr_tot = cat(4, dl_corr_tot, dl_hcp);
    ls_corr_tot = cat(4, ls_corr_tot, ls_hcp);
    test_maps = cat(4, test_maps, diffmap);

    test_metrics(sub, :) = [sub, counts.countDL, counts.countLS, counts.tot];

    fprintf(string(datetime) + '\n  --  Finished sub %d \n', sub);
    toc;
    clear HCP LS DL3d_recon
    fprintf('\nNumber of improved for subject %d: %i/%i --> %.1f percent \n', sub, counts.countDL, counts.tot, counts.countDL/counts.tot*100)
end


%% Coeffmap
diffmap = dl_corr_tot - ls_corr_tot;
four_maps = [];
for i = 1:15
    if i == 1 || mod(i, 5) == 0
        thissubj = diffmap(:,:,:,i);
        this_map = cat(1, rot90(thissubj), rot90(permute(thissubj, [3, 2, 1]), 2), rot90(permute(thissubj, [3, 1, 2]), 2));
        four_maps = cat(2, four_maps, this_map);
    end
end
%% Plot

smoothim = smooth3(four_maps, 'gaussian', 3);
h = implay(mat2gray(smoothim));
h.Visual.ColorMap.Map = parula;
smoothslice = smoothim(:,:,68);
h2 = imshow(smoothslice, [-0.5, 0.5],'Colormap', parula(256));
colorbar
%% Plot bar graph

percentDL = test_metrics(:,2)./test_metrics(:,4)*100;
percentLS = test_metrics(:,3)./test_metrics(:,4)*100;
percent = cat(1, percentLS', percentDL');
b = bar(percent','stacked', 'FaceColor','flat');
b(1).CData = [0 0.4470 0.7410];
b(2).CData = [0.9290 0.9 0];
xtips1 = b(2).XEndPoints;
ytips1 = b(2).YEndPoints;
labels1 = string(round(b(2).YData));
for i = 1:length(b(2).YData)
    labels1(i) = string(round(percentDL(i))) + '%';
end
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','top')
title('Proportion of improved voxel correlations')
xlabel('Participant number')
set(b, {'DisplayName'}, {'LS > 3D-UNet','3D-UNet > LS'}')
legend()
ylabel('Percentage')
Ax = gca;
Ax.Box = 'off';

