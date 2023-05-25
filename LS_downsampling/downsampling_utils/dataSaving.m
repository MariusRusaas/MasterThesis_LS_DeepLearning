function isFinished = dataSaving(savePath, subnumber, orig_data, savedata_LS, savedata_k)
%SAVING Summary of this function goes here
%   Detailed explanation goes here

LS_savepath = [savePath, '\', 'LS_imgs', '\', subnumber, '_rfMRI_LS_proc'];
niftiwrite(savedata_LS, LS_savepath);

orig_savepath = [savePath, '\', 'Original_HCP', '\', subnumber, '_rfMRI_HCP_proc'];
niftiwrite(orig_data, orig_savepath);

k_savepath = [savePath, '\', 'k_downsampled_processed', '\', subnumber, '_downsampled_k_proc.csv'];
writematrix(savedata_k, k_savepath)

isFinished = true;
end

