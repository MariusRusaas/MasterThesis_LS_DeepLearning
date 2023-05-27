function [image,filename, info] = findImage_processed(pathname, subnumber)
%FINDIMAGE_PROCESSED 
%   function that loads the image from a processed HCP rfMRI subject folder
%   with the help of the pathTraverse(...) function.

%   Input:
%   imagefolder -   name of path for a rfMRI folder
%   Subnumber   -   Subject number string

%   Output:
%   image       -   struct with 4-D fMRI image and dimension + fov data
%   filename    -   name of nii-file

filepath = pathTraverse(pathname, subnumber);
name_split = split(filepath, '\');
filename = cell2mat(name_split(end));

info = niftiinfo(filepath);
im = niftiread(filepath); im = im2single(im);
image.img = im; image.dim = size(image.img);
image.pixdim = info.PixelDimensions;
image.fov = image.dim.*image.pixdim;
disp('Subject loaded')
end


function [filepath] = pathTraverse(pathname, subnumber)
%FILETRAVRESE function that finds the filename of resting state fMRI data
%from unzipped subject folder of HCP project.
%   Input:
%   pathname    - unzipped subject folder
%   Subnumber   - Sibject id as string

%   Output:
%   filepath    - total path of the relevant fMRI data nii-file

foldername = [pathname, '\', subnumber, '\MNINonLinear\Results\rfMRI_REST1_7T_PA'];
if ~exist(foldername, 'dir') == 0
    if exist([foldername, '\rfMRI_REST1_7T_PA.nii']) == 2
        filepath = [foldername, '\rfMRI_REST1_7T_PA.nii'];
    elseif exist([foldername, '\rfMRI_REST1_7T_PA.nii.gz']) == 2
        filepath = [foldername, '\rfMRI_REST1_7T_PA.nii.gz'];
        %filepath = fullfile(cell2mat(filename)); 
    end
else, filepath = 'Does not exist';
end
   
end

