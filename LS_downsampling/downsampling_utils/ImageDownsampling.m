function [orig_fin, LS_fin, k_tot] = ImageDownsampling(GmriObj, DCF, image, n_imgs, image_size, saving_size, num_slices)
%IMAGEDOWNSAMPLING Summary of this function goes here
%   Detailed explanation goes here
arguments
    GmriObj;
    DCF;
    image;
    n_imgs(1,1) single {mustBeFinite} = size(image, 4);
    image_size(1,1) single {mustBeFinite} = 150;
    saving_size(1,1)single {mustBeFinite} = 130;
    num_slices(1,1)single {mustBeFinite} = 90;
end

LS_tot = [];
orig_tot = [];
k_tot = [];

% Loading images in folder
for j = 1 : n_imgs
    % Square image and increase image size
    if image_size == 0, sqrd_image = ImageSquareing(image.img(:,:,:,j));
    else, sqrd_image = ImageSquareing(image.img(:,:,:,j), image_size); end
    % Update dims if changed
    dim = size(sqrd_image);
    % creating saving dims
    tot = dim(1)-saving_size;
    left = floor(tot/2) + 1; right = dim(1)-ceil(tot/2);
    slice_rem = dim(3)-num_slices;
    left_slice = floor(slice_rem/2) + 1; right_slice = dim(3)-ceil(slice_rem/2);
    
    % Conjugate reconstruction of image:
    mask = true(size(sqrd_image));
    [LS_image, k_samples] = GmriRecon(GmriObj, sqrd_image, DCF, mask);
    if j == 1 || j == n_imgs || mod(j, 50) == 0 , fprintf('Volume %d / %d \n', j, n_imgs); end
    
    LS_tot = cat(4, LS_tot, LS_image(left:right, left:right, left_slice:right_slice));
    orig_tot = cat(4, orig_tot, sqrd_image(left:right, left:right, left_slice:right_slice));
   
    k_tot = cat(2, k_tot, k_samples);
end
% LS_tot = LS_tot(left:right, left:right, left_slice:right_slice,:);
% orig_tot = orig_tot(left:right, left:right, left_slice:right_slice,:);

% Normalizing
orig_norm = orig_tot - min(orig_tot(:)); orig_fin = orig_norm ./ max(orig_norm(:));
LS_norm = LS_tot - min(LS_tot(:)); LS_fin = LS_norm ./ max(LS_norm(:));

end


function [reconImage, k_samples] = GmriRecon(Am, image, DCF, mask)
%GMRIRECON:
%   Small method that performs the forward and backward operations of the
%   Gmri object (ref MIRT by J. Fessler). The method include basis effect
%   mitigation.
%	y = Am * x		forward operation
%	x = Am' * y		adjoint operation
%
%   Input:
%   Am      - Gmri object
%   image   - image matrix (3D single)
%   DCF     - Density correctino factors for k-space trajectory
%   Mask    - Mask with same dimension as image for reconstruction filling
%
%   Output:
%   reconImage - 3D single reconstructed image matrix

wi_basis = DCF ./ Am.arg.basis.transform; % trick! undo basis effect

%First we need to make the k-space samples:
k_samples = Am * image;

cp_image = Am' * (wi_basis .* k_samples);
reconImage = abs(embed(cp_image, mask));
end
