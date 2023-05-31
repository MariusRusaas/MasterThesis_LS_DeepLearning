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

sub = 1;

HCP = niftiread([filelist_orig(sub).folder, '\', filelist_orig(sub).name]);
HCP_vol = HCP(:,:,:,1);

LS = niftiread([filelist_LS(sub).folder, '\', filelist_LS(sub).name]);
LS_vol = LS(:,:,:,1);

DL3d_recon = niftiread([filelist_DL3d(sub).folder, '\', filelist_DL3d(sub).name]);
DL3d_vol = DL3d_recon(:,:,:,1);

%% Subplotting activation

x = 1:900; num = 1:4; x_ax = 64; y_ax = 42; z_ax = 60;

pcc_corr = zeros(4, 6);

figure
for i = num
    subplot(2,2,i);
    if i > length(num)/2, col = length(num)/2; row = 1;
    else, col = 0; row = 0; end
    HCP_vox = squeeze(HCP(x_ax-col+(i-1), y_ax+row, z_ax, x)); 
    LS_vox = squeeze(LS(x_ax-col+(i-1), y_ax+row, z_ax, x)); 
    DL_vox = squeeze(DL3d_recon(x_ax-col+(i-1), y_ax+row, z_ax, x)); 

    ft_HCP = fft(HCP_vox); ft_LS = fft(LS_vox); ft_DL = fft(DL_vox);
    ft_HCP(1:6) = 0; ft_LS(1:6) = 0; ft_DL(1:6) = 0;
    ft_HCP(894:900) = 0; ft_LS(894:900) = 0; ft_DL(894:900) = 0;
    ft_HCP(83:818) = 0; ft_LS(83:818) = 0; ft_DL(83:818) = 0;
    HCP_voxn = ifft(ft_HCP); LS_voxn = ifft(ft_LS); DL_voxn = ifft(ft_DL);
    HCP_voxn = real(HCP_voxn) + imag(HCP_voxn);LS_voxn = real(LS_voxn) + imag(LS_voxn); DL_voxn = real(DL_voxn) + imag(DL_voxn);



    hold on
    movm = 10;
    plot(x,movmean(HCP_voxn, movm), 'color', [.6 .6 .6]);
    plot(x,movmean(LS_voxn, movm) , 'LineWidth' ,1.1, 'color', '#0072BD');
    plot(100, 50, '-*', 'MarkerSize',3, 'color', '#D95319')
    plot(x, movmean(DL_voxn, movm), 'color', '#D95319');
    DL_mov = movmean(DL_voxn, movm);
    plot(x(1:5:end), DL_mov(1:5:end), '*', 'MarkerSize',2, 'color', '#D95319')

    legend('HCP', 'LS', '3D-UNet', 'Location', 'northwest')
    xlabel('Time [s]')
    ylabel('Voxel variation')
    axis([x(1), x(length(x)), -0.03, 0.03])
    grid on
    hold off

    LS_corr = corrcoef(LS_voxn, HCP_voxn);DL_corr = corrcoef(DL_voxn, HCP_voxn);DL_corrL = corrcoef(DL_voxn, LS_voxn);
    LS_sim = sum(LS_voxn.*HCP_voxn); DL_sim = sum(DL_voxn.*HCP_voxn); DL_simL = sum(DL_voxn.*LS_voxn);

    pcc_corr(i, :) = [LS_corr(1,2), DL_corr(1,2), DL_corrL(1,2), LS_sim, DL_sim, DL_simL];
    
end
sgtitle('Voxel variations of four neighboring voxels in the posterior cingulate cortex')

%% Subplotting unactivation

x = 1:900;num = 1:4;x_ax = 90;y_ax = 85;z_ax = 50;

fl_corr = zeros(4, 6);

figure
for i = num
    subplot(2,2,i);
    if i > length(num)/2, col = length(num)/2; row = 1;
    else, col = 0; row = 0; end
    HCP_vox = squeeze(HCP(x_ax-col+(i-1), y_ax+row, z_ax, x)); 
    LS_vox = squeeze(LS(x_ax-col+(i-1), y_ax+row, z_ax, x)); 
    DL_vox = squeeze(DL3d_recon(x_ax-col+(i-1), y_ax+row, z_ax, x));  

    ft_HCP = fft(HCP_vox); ft_LS = fft(LS_vox); ft_DL = fft(DL_vox);
    ft_HCP(1:6) = 0; ft_LS(1:6) = 0; ft_DL(1:6) = 0;
    ft_HCP(894:900) = 0; ft_LS(894:900) = 0; ft_DL(894:900) = 0;
    ft_HCP(83:818) = 0; ft_LS(83:818) = 0; ft_DL(83:818) = 0;
    HCP_voxn = ifft(ft_HCP); LS_voxn = ifft(ft_LS); DL_voxn = ifft(ft_DL);
    HCP_voxn = real(HCP_voxn) + imag(HCP_voxn);LS_voxn = real(LS_voxn) + imag(LS_voxn); DL_voxn = real(DL_voxn) + imag(DL_voxn);
    
    hold on
    movm= 10;
    plot(x,movmean(HCP_voxn, movm), 'color', [.6 .6 .6]);
    plot(x,movmean(LS_voxn, movm) , 'LineWidth' ,1.1, 'color', '#0072BD');
    plot(100, 50, '-*', 'MarkerSize',3, 'color', '#D95319')
    plot(x, movmean(DL_voxn, movm), 'color', '#D95319');
    DL_mov = movmean(DL_voxn, movm);
    plot(x(1:5:end), DL_mov(1:5:end), '*', 'MarkerSize',2, 'color', '#D95319')
    legend('HCP', 'LS', '3D-UNet', 'Location', 'northwest')
    xlabel('Time [s]')
    ylabel('Voxel variation')
    axis([x(1), x(length(x)), -0.015, 0.015])
    grid on
    hold off

    LS_corr = corrcoef(LS_voxn, HCP_voxn);DL_corr = corrcoef(DL_voxn, HCP_voxn);DL_corrL = corrcoef(DL_voxn, LS_voxn);
    LS_sim = sum(LS_voxn.*HCP_voxn); DL_sim = sum(DL_voxn.*HCP_voxn); DL_simL = sum(DL_voxn.*LS_voxn);

    fl_corr(i, :) = [LS_corr(1,2), DL_corr(1,2), DL_corrL(1,2), LS_sim, DL_sim, DL_simL];
end
sgtitle('Voxel variations of four neighboring voxels in the frontal lobe')

%% Coeffmap?
counts.countDL = 0;
counts.countLS = 0;
counts.tot = 0;
dims = size(DL3d_vol);
ls_hcp = zeros(dims(1:3));
dl_ls = zeros(dims(1:3));
dl_hcp = zeros(dims(1:3));
diffmap = zeros(dims(1:3));
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

                ft_HCP = fft(HCP_vox); ft_LS = fft(LS_vox); ft_DL = fft(DL_vox);
                ft_HCP(1:6) = 0; ft_LS(1:6) = 0; ft_DL(1:6) = 0;
                ft_HCP(894:900) = 0; ft_LS(894:900) = 0; ft_DL(894:900) = 0;
                ft_HCP(83:818) = 0; ft_LS(83:818) = 0; ft_DL(83:818) = 0;
                HCP_voxn = ifft(ft_HCP); LS_voxn = ifft(ft_LS); DL_voxn = ifft(ft_DL);
                HCP_voxn = real(HCP_voxn) + imag(HCP_voxn);LS_voxn = real(LS_voxn) + imag(LS_voxn); DL_voxn = real(DL_voxn) + imag(DL_voxn);

                LS_corr = corrcoef(LS_voxn, HCP_voxn);
                DL_corrH = corrcoef(DL_voxn, HCP_voxn);
                DL_corr = corrcoef(DL_voxn, LS_voxn);

                diffmap(row, col, slice) = DL_corrH(1, 2) - LS_corr(1, 2);
                ls_hcp(row, col, slice) = LS_corr(1, 2);
                dl_hcp(row, col, slice) = DL_corrH(1, 2);
                dl_ls(row, col, slice) = DL_corr(1, 2);
            end
        end
    end
    prog = ( 100*(slice/dims(3)) );
    fprintf(1,'\b\b\b\b%3.0f%%',prog);
end
fprintf('\n')
%%
diff1 = rot90(permute(diffmap, [3, 2, 1]), 2);
diff2 = rot90(permute(diffmap, [3, 1, 2]), 2);
%sqdiff = cat(3, zeros(128,128,7), diffmap, zeros(128,128,8));
diff1 = cat(1, diffmap, diff1(:,:,8:120), diff2(:,:,8:120));

h2 = implay(mat2gray(diff1));
h2.Visual.ColorMap.Map = parula;

%%
diffmap2 = dl_hcp - ls_hcp;

diffg1 = rot90(permute(diffmap2, [3, 2, 1]), 2);
diffg2 = rot90(permute(diffmap2, [3, 1, 2]), 2);
diffg1 = cat(1, diffmap2, diffg1(:,:,8:120), diffg2(:,:,8:120));

h2 =implay(mat2gray(diffg1, [-0.5, 0.5]));
h2.Visual.ColorMap.Map = parula;


