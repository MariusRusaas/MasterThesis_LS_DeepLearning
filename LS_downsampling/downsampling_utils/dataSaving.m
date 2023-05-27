function isFinished = dataSaving(savePath, subnumber, orig_data, savedata_LS, savedata_k)
%DATASAVING:
%   Method that saves the downsampled Looping Star image, original HCP data
%   (with normalization, and new image dimensions from squareing), and
%   k-space samples in an organized path.
%   
%   Input:
%   savePath    - dataset path
%   subnumber   - Subject ID
%   orig_data   - Image matrix of original HCP data
%   savedata_LS - Image matrix of Looping Star image
%   savedata_k  - downsampled k-space samples
%   
%   Output:
%   isFinished - Boolean that indicates process success

LS_savepath = [savePath, '\', 'LS_imgs', '\', subnumber, '_rfMRI_LS_proc'];
niftiwrite(savedata_LS, LS_savepath);

orig_savepath = [savePath, '\', 'Original_HCP', '\', subnumber, '_rfMRI_HCP_proc'];
niftiwrite(orig_data, orig_savepath);

k_savepath = [savePath, '\', 'k_downsampled_processed', '\', subnumber, '_downsampled_k_proc.csv'];
writematrix(savedata_k, k_savepath)

isFinished = true;
end

