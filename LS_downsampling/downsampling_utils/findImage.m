function [image,filename, info] = findImage(pathname, subnumber)
%FINDIMAGE 
%   function that loads the image from an HCP rfMRI subject folder
%   with the help of the pathTraverse(...) function. 

%   Input:
%   imagefolder -   name of path for a rfMRI folder
%   Subnumber   -   Subjet number string

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
%   Subnumber   - Subject id as string

%   Output:
%   filepath    - total path of the relevant fMRI data nii-file

foldername = [pathname, '\', subnumber, '\unprocessed\7T\rfMRI_REST1_PA'];
if ~exist(foldername, 'dir') == 0
    if exist([foldername, '\', subnumber, '_7T_rfMRI_REST1_PA.nii']) == 2
        filepath = [foldername, '\', subnumber, '_7T_rfMRI_REST1_PA.nii'];
    elseif exist([foldername, '\', subnumber, '_7T_rfMRI_REST1_PA.nii.gz']) == 2
        filepath = [foldername, '\', subnumber, '_7T_rfMRI_REST1_PA.nii.gz'];
        %filepath = fullfile(cell2mat(filename)); 
    end
else, filepath = 'Does not exist';
end
   
end

