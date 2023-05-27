%% Script that calculates the MSE, SSIM, PSNR and SNR for the unprocessed dataset

clear

orig_dir = 'E:\Master_Project_Marius\Unprocessed_HCP_data\Testing_data\Original_fMRI';
LS_dir = 'E:\Master_Project_Marius\Unprocessed_HCP_data\Testing_data\LS_imgs';
DL2D_dir = 'E:\Master_Project_Marius\Unprocessed_HCP_data\Testing_data\DL_recon\2D-UNet_3aug';
DL3D_dir = 'E:\Master_Project_Marius\Unprocessed_HCP_data\Testing_data\DL_recon\3D-UNet';


filelist_LS = dir(LS_dir);
filelist_LS = filelist_LS(3:end);
filelist_orig = dir(orig_dir);
filelist_orig = filelist_orig(3:end);
filelist_DL2d = dir(DL2D_dir);
filelist_DL2d = filelist_DL2d(3:end);
filelist_DL3d = dir(DL3D_dir);
filelist_DL3d = filelist_DL3d(3:end);

leng = length(filelist_orig);

%% Calculating metrics

test_metrics_arr = zeros(15, 13);
for i = 1:leng
    LS = niftiread([filelist_LS(i).folder, '\', filelist_LS(i).name]);
    LS = LS(2:129,2:129,:,:);
    LS_vol = LS(:,:,:,1);
    HCP = niftiread([filelist_orig(i).folder, '\', filelist_orig(i).name]);
    HCP = HCP(2:129,2:129,:,:);
    HCP_vol = HCP(:,:,:,1);
    DL2d_recon = niftiread([filelist_DL2d(i).folder, '\', filelist_DL2d(i).name]);
    DL2d_vol = DL2d_recon(:,:,:,1);
    DL3d_recon = niftiread([filelist_DL3d(i).folder, '\', filelist_DL3d(i).name]);
    DL3d_vol = DL3d_recon(:,:,:,1);
    
    test_metrics_arr(i, 1:3) = [immse(HCP, LS), immse(HCP, DL2d_recon), immse(HCP, DL3d_recon)];
    test_metrics_arr(i, 4:6) = [ssim(LS_vol, HCP_vol),  ssim(DL2d_vol, HCP_vol),  ssim(DL3d_vol, HCP_vol)];
    test_metrics_arr(i, 7:9) = [psnr(LS, HCP),  psnr(DL2d_recon, HCP),  psnr(DL3d_recon, HCP)];
    test_metrics_arr(i, 10:13) = [SnR(HCP_vol), SnR(LS_vol),  SnR(DL2d_vol),  SnR(DL3d_vol)];
    
    fprintf('MSE for subject %d: %4f, %4f, %4f, %.3f, %.3f, %.3f \n', i, test_metrics_arr(i, 1), test_metrics_arr(i, 2), test_metrics_arr(i, 3), test_metrics_arr(i, 4), test_metrics_arr(i, 5), test_metrics_arr(i, 6))

end

%% Summarizing raw metrics
HCP_snr = mean(test_metrics_arr(:,10)); HCP_snr_std = std(test_metrics_arr(:,10));

LS_mse = mean(test_metrics_arr(:,1)); LS_mse_std = std(test_metrics_arr(:,1));
LS_ssim = mean(test_metrics_arr(:,4)); LS_ssim_std = std(test_metrics_arr(:,4));
LS_psnr = mean(test_metrics_arr(:,7)); LS_psnr_std = std(test_metrics_arr(:,7));
LS_snr = mean(test_metrics_arr(:,11)); LS_snr_std = std(test_metrics_arr(:,11));

DL2d_mse = mean(test_metrics_arr(:,2)); DL2d_mse_std = std(test_metrics_arr(:,2));
DL2d_ssim = mean(test_metrics_arr(:,5)); DL2d_ssim_std = std(test_metrics_arr(:,5));
DL2d_psnr = mean(test_metrics_arr(:,8)); DL2d_psnr_std = std(test_metrics_arr(:,8));
DL2d_snr = mean(test_metrics_arr(:,12)); DL2d_snr_std = std(test_metrics_arr(:,12));

DL3d_mse = mean(test_metrics_arr(:,3)); DL3d_mse_std = std(test_metrics_arr(:,3));
DL3d_ssim = mean(test_metrics_arr(:,6)); DL3d_ssim_std = std(test_metrics_arr(:,6));
DL3d_psnr = mean(test_metrics_arr(:,9)); DL3d_psnr_std = std(test_metrics_arr(:,9));
DL3d_snr = mean(test_metrics_arr(:,13)); DL3d_snr_std = std(test_metrics_arr(:,13));

test_metrics_struct = [0       , 0           , 0        , 0            , 0        , 0            , HCP_snr , HCP_snr_std;
                       LS_mse  , LS_mse_std  , LS_ssim  , LS_ssim_std  , LS_psnr  , LS_psnr_std  , LS_snr  , LS_snr_std  ;
                       DL2d_mse, DL2d_mse_std, DL2d_ssim, DL2d_ssim_std, DL2d_psnr, DL2d_psnr_std, DL2d_snr, DL2d_snr_std;
                       DL3d_mse, DL3d_mse_std, DL3d_ssim, DL3d_ssim_std, DL3d_psnr, DL3d_psnr_std, DL3d_snr, DL3d_snr_std];

test_metrics_round = [round(test_metrics_arr(:,1), 4), round(test_metrics_arr(:,2:3), 5), round(test_metrics_arr(:,4:6), 3), round(test_metrics_arr(:,7:9), 2), round(test_metrics_arr(:,10:13), 2)];
