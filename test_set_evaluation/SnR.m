function snr = SnR(volume)
%SNR:
%   This method calculates the signal_to_noise ratio in an input volume.
%   The SnR is calculated using this equation:
%
%       SnR = mean(S)/std(N)
%
%   Here, S is a large area of the image with signal from the brain. N is
%   regions in the image with no signal from the brain (air).

%   Input:
%   Volume  - 3D image of a brain
%   
%   Output:
%   snr     - Numerical signal-to-noise ratio 
    
    dims = size(volume);
    N_area_1 = volume(2:11,2:11,5:14);
    
    N_area_2 = volume(dims(1)-11:dims(1)-2,dims(2)-11:dims(2)-2,6:15);
    N_area_3 = volume(2:11,dims(2)-11:dims(2)-2,dims(3)-14:dims(3)-5);
    N_area_4 = volume(2:11,dims(2)-11:dims(2)-2,dims(3)-14:dims(3)-5);
    
    N_tot = cat(2, N_area_1, N_area_2, N_area_3, N_area_4);
    
    N_std = std(N_tot(:));

    mid_slice = floor(dims(3)/2);
    sig = volume(50:80, 50:80, mid_slice-10:mid_slice+10);

    snr =  mean(sig(:)) / (N_std + 1e-6);
end

