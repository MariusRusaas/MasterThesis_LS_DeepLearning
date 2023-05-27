%% This is a script that performs the downsampling of the training and testing set
clear
% k-space trajectory
load('2echoes_kspace.mat')
% Run MIRT setup
addpath 'C:\Users\MariusER\Documents\GitHub\Master-Project-LoopingStar\MIRT\irt';
run('C:\Users\MariusER\Documents\GitHub\Master-Project-LoopingStar\MIRT\irt\setup.m');

%% Create list of all zipped folders
unzippedpath = 'E:\Master_Project_Marius\Processed_HCP_data\unzipped';
savepath_train = 'E:\Master_Project_Marius\Processed_HCP_data\Training_data';
savepath_test = 'E:\Master_Project_Marius\Processed_HCP_data\Testing_data';
filelist = dir('E:\Master_Project_Marius\Processed_HCP_data\Processed_dwnld_2\*.zip');

load("test_subj.mat");

%% Fetching data from file and loading image

[image, filename, info] = findImage_processed(unzippedpath, '100610');

dims = size(image.img);

%% Conjugate phase reconstruction of HCP image
% Creating Gmri object:
%--------------------------------------------------------------------------
% Reconstruction with smaller image size result in less smoothing of the
% image, but also result in some ghosting artefacts. A Image size of 150
% retain a good amount of spatial resolution while mainly pushing the 
% artefact out of the head image in unprocessed data
% For processed data, a image size of 170 reduce artifacts.
%--------------------------------------------------------------------------
image_size = 170;
N = repmat(image_size, [1, 3]);
nufft_args = {N, 6*ones(size(N)), 2*N, N/2, 'table', 2^12, 'minmax:kb'};
mask = true(N);
clear N
basis = {'rect'};
Am = Gmri(test_coords, mask, ...
		'fov', image.fov(1:3), 'basis', basis, 'nufft', nufft_args);
disp('Gmri object ready.')

%% Downsampling several images
fprintf("\nStarting downsampling and saving: \n")

for i = 1:length(filelist)
    subnumber = filelist(i).name(1:6);

    tic;
   
    if ~any(strcmp(test_subj,subnumber)) % Training set subjects
        % Extracting image content
        [image, filename, info] = findImage_processed(unzippedpath, subnumber);

        % Downsampling image!
        [orig_image, LS_image, k_samps] = ImageDownsampling(Am, DCF, ...
                                               image, 50, image_size, dims(2), dims(1));
        %!! - normalized - !!

        done = dataSaving(savepath_train, subnumber, orig_image, LS_image, k_samps);
        toc;
        fprintf('Subject %d / %d done! Training Subject: %s \n', i, length(filelist), subnumber);

    elseif any(strcmp(test_subj, subnumber))  % Testing set subjects
        % Extracting image content
        [image, filename, info] = findImage_processed(unzippedpath, subnumber);
        
        % Downsampling image!
        [orig_image, LS_image, k_samps] = ImageDownsampling(Am, DCF, ...
                                               image, 900, image_size, 128, dims(1));
        
        %!! - normalized - !!

        done = dataSaving(savepath_test, subnumber, orig_image, LS_image, k_samps);
        toc;
        fprintf('Subject %d / %d done! Testing Subject: %s \n', i, length(filelist), subnumber);
        
    else, fprintf('Subject %d / %d skipped \n', i, length(filelist)); 
    end
end

%% Have a look at last subject
movie1 = cat(2, orig_image(:,:,:,1), LS_image(:,:,:,1)); 
movie2 = cat(2, orig_image(:,:,:,2), LS_image(:,:,:,2));
movie = cat(1, movie1, movie2);


%% implay
implay(mat2gray(movie));
