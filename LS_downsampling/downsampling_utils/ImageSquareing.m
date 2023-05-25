function [square_image] = ImageSquareing(inputImage, in_dim)
%IMAGESQUAREING:
%   Method that increase increase the matrix dimensions of a rectangular
%   image by adding zero 2D matrices to the edges of the smallest
%   dimensions to fit the largest dimension.
%   
%   Input:
%   inputImage - Non-square image
%   
%   Output:
%   Square_image - squared and upsized image

%default:
arguments
    inputImage;
    in_dim(1,1) single {mustBeFinite} = max(size(inputImage));
end
if in_dim < max(size(inputImage)), in_dim = max(size(inputImage)); end
max_dim = in_dim;

dim = size(inputImage);
if dim(1) < max_dim
    dim_toadd = max_dim - dim(1);
    front = ceil(dim_toadd/2); back = floor(dim_toadd/2);
    front_cat = zeros(front, dim(2),dim(3)); back_cat = zeros(back, dim(2),dim(3)); 
    inputImage = cat(1, front_cat, inputImage, back_cat);
end, dim = size(inputImage); 
if dim(2) < max_dim
    dim_toadd = max_dim - dim(2);
    front = ceil(dim_toadd/2); back = floor(dim_toadd/2);
    front_cat = zeros(dim(1), front, dim(3)); back_cat = zeros(dim(1), back ,dim(3)); 
    inputImage = cat(2, front_cat, inputImage, back_cat);
end, dim = size(inputImage);
if dim(3) < max_dim
    dim_toadd = max_dim - dim(3);
    front = ceil(dim_toadd/2); back = floor(dim_toadd/2);
    front_cat = zeros(dim(1), dim(2), front); back_cat = zeros(dim(1), dim(2) , back); 
    inputImage = cat(3, front_cat, inputImage, back_cat);
end

square_image = inputImage;
end

