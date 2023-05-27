import os
import numpy as np
import nibabel as nib
import random
import torch as t
from torch.utils.data import Dataset

#################################################################################################################################################

class LS_dataset_subj(Dataset):
    """This Dataset class loads pairs of Looping Star images and HCP images into the computer memory. The entire subject file is loaded.

    Parameters:
    - label_path (str): Path to the NIfTI-files containing HCP data
    - input_path (str): Path to the NIfTI-files containing LS data

    Returns:
    - Tensor: Torch tensor with LS data
    - Tensor: Torch tensor with HCP data

   """
    def __init__(self, label_path, input_path):
        self.label_path = label_path
        self.input_path = input_path

        self.num_subjects = len(os.listdir(self.label_path))

    def __len__(self):
        return self.num_subjects

    def __getitem__(self, index):
        orig_listdir = os.listdir(self.label_path)
        orig_file = sorted(orig_listdir)[index]
        LS_listdir = os.listdir(self.input_path)
        LS_file = sorted(LS_listdir)[index]
        if not orig_file[0:6] == LS_file[0:6]: print('Mismatching subjects: LS = {} | Orig = {}'.format(LS_file[0:6], orig_file[0:6]))
        
        f_orig = os.path.join(self.label_path, orig_file)
        f_LS = os.path.join(self.input_path, LS_file)

        orig_tot = nib.load(f_orig).get_fdata()
        LS_tot = nib.load(f_LS).get_fdata()

        if LS_tot.shape[0] > 128 and LS_tot.shape[0] == LS_tot.shape[1]:
            excess = (LS_tot.shape[0]-128)//2
            LS_tot = LS_tot[excess:-excess, excess:-excess,:]
            orig_tot = orig_tot[excess:-excess, excess:-excess,:]

        
        return t.from_numpy(LS_tot.copy()), t.from_numpy(orig_tot.copy())


#################################################################################################################################################

class LS_dataset_VolperSubj(Dataset):
    """This Dataset class fetches a training pair consisting of one Looping Star volume and one HCP volume
    from the subject data in the comuter memory for training 3D models. The data can either be loaded as the images are in the memory
    or augmented by rotation or flipping.

    Parameters:
    - input_vols (tensor):  Complete Looping Star image data for all subjects in the computer memoy
    - label_vols (tensor):  Complete HCP image data for all subjects in the computer memoy
    - num_vol (int):        Number of volumes per subject to include in training (Default = None, considers all possible volumes)
    - num_augment (int):    How many augmentation possibilities to include, Can be between 0 and 2 (Default = 0)

    Returns:
    - Tensor: Torch tensor with one LS volume
    - Tensor: Torch tensor with one HCP volume

   """
    def __init__(self, input_vols, label_vols, num_vol = None, num_augment = 0):
        self.input_vols = input_vols
        self.label_vols = label_vols

        inp_shp = self.input_vols.shape
        lbl_shp = self.label_vols.shape
        inp_vol = inp_shp[4]
        lbl_vol = lbl_shp[4]
        if num_vol == None:
            self.num_vols = min(inp_vol, lbl_vol)
        else:
            self.num_vols = num_vol

        self.num_subs = inp_shp[0]
        self.num_aug = num_augment

    def __len__(self):
        return (self.num_aug+1)*self.num_subs*self.num_vols

    def get_aug_sub_vol_idx(self, index):
        aug_idx = index // (self.num_subs*self.num_vols)
        sub_idx = (index - aug_idx*self.num_subs*self.num_vols) // (self.num_vols)
        vol_idx = (index - aug_idx*self.num_subs*self.num_vols - sub_idx*self.num_vols) % self.num_vols

        return aug_idx, sub_idx, vol_idx

    def __getitem__(self, index):
        aug_idx, sub_idx, vol_idx = self.get_aug_sub_vol_idx(index)

        orig_vol = self.label_vols[sub_idx,:,:,:,vol_idx]
        LS_vol = self.input_vols[sub_idx,:,:,:,vol_idx]

        if aug_idx == 1: 
            rot_num = random.randint(1,3)
            orig_vol_out = t.rot90(orig_vol, k=rot_num, dims = [0, 1])
            LS_vol_out = t.rot90(LS_vol, k=rot_num, dims = [0, 1])
        elif aug_idx == 2:
            fl_num = random.randint(0,1)
            if fl_num == 0:
                orig_vol_out = t.flip(orig_vol, dims=[1,2])
                LS_vol_out = t.flip(LS_vol, dims=[1,2])
            else:
                orig_vol_out = t.flip(orig_vol, dims=[0,2])
                LS_vol_out = t.flip(LS_vol, dims=[0,2])
        else:
            return t.unsqueeze(LS_vol, 0), t.unsqueeze(orig_vol, 0)

        return t.unsqueeze(LS_vol_out, 0), t.unsqueeze(orig_vol_out, 0)
    

#################################################################################################################################################

class LS_dataset_SliceperSubj(Dataset):
    """This Dataset class fetches a training pair consisting of one Looping Star and one HCP image slice
    from the subject data in the comuter memory for training 2D models. The data can either be loaded as the images are in the memory
    or augmented by rotation.

    Parameters:
    - input_vols (tensor):  Complete Looping Star image data for all subjects in the computer memoy
    - label_vols (tensor):  Complete HCP image data for all subjects in the computer memoy
    - num_vol (int):        Number of volumes per subject to include in training (Default = None, considers all possible volumes)
    - num_slices (int):        Number of slices per volume to include in training (Default = None, considers all possible volumes)
    - num_augment (int):    How many augmentation possibilities to include, Can be between 0 and 2 (Default = 0)

    Returns:
    - Tensor: Torch tensor with one LS image slice
    - Tensor: Torch tensor with one HCP image slice
    
   """
    def __init__(self, input_vols, label_vols, num_vol = None, num_slices = None, num_augment = 0):
        self.input_vols = input_vols
        self.label_vols = label_vols
        
        inp_shp = self.input_vols.shape
        lbl_shp = self.label_vols.shape
        inp_vol = inp_shp[4]
        lbl_vol = lbl_shp[4]
        if num_vol == None:
            self.num_vols = min(inp_vol, lbl_vol)
        else:
            self.num_vols = num_vol

        inp_slice = inp_shp[3]
        lbl_slice = lbl_shp[3]
        if num_slices == None:
            self.num_slices = min(inp_slice, lbl_slice)
        else:
            self.num_slices = num_vol

        self.num_subs = inp_shp[0]
        self.num_aug = num_augment

    def __len__(self):
        return (self.num_aug+1)*self.num_subs*self.num_vols*self.num_slices

    def get_aug_sub_vol_slice_indx(self, index):
        aug_idx = index // (self.num_subs*self.num_vols*self.num_slices)
        sub_idx = (index - aug_idx*self.num_subs*self.num_vols*self.num_slices) // (self.num_vols*self.num_slices)
        vol_idx = (index - aug_idx*self.num_subs*self.num_vols*self.num_slices - sub_idx*self.num_vols*self.num_slices) // (self.num_slices)
        slice_idx = (index - aug_idx*self.num_subs*self.num_vols*self.num_slices - sub_idx*self.num_vols*self.num_slices - vol_idx*self.num_slices) % self.num_slices

        return aug_idx, sub_idx, vol_idx, slice_idx

    def __getitem__(self, index):
        aug_idx, sub_idx, vol_idx, slice_idx = self.get_aug_sub_vol_slice_indx(index)

        orig_slice = self.label_vols[sub_idx,:,:,slice_idx, vol_idx]
        LS_slice = self.input_vols[sub_idx,:,:,slice_idx, vol_idx]

        if aug_idx == 1: 
            rot_num = random.choice([1,3])
            orig_slice_out = t.rot90(orig_slice, k=rot_num, dims = [0, 1])
            LS_slice_out = t.rot90(LS_slice, k=rot_num, dims = [0, 1])
        elif aug_idx == 2:
            orig_slice_out = t.rot90(orig_slice, k=2, dims = [0, 1])
            LS_slice_out = t.rot90(LS_slice, k=2, dims = [0, 1])
        else:
            return t.unsqueeze(LS_slice, 0), t.unsqueeze(orig_slice, 0)

        return t.unsqueeze(LS_slice_out, 0), t.unsqueeze(orig_slice_out, 0)


#################################################################################################################################################

class LS_dataset_subjSave(Dataset):
    """This Dataset class loades LS data from one subject. The class is used when reconstructing images from test set subjects.
    The reconstructed data is to be saved in NIfTI-files for further processing. 

    Parameters:
    - input_path (str): Path to the NIfTI-files containing LS data

    Returns:
    - Tensor: Torch tensor with LS data
    - String: Subject ID
    
   """
    def __init__(self, input_path):
        self.input_path = input_path

        self.num_subjects = len(os.listdir(self.input_path))

    def __len__(self):
        return self.num_subjects

    def __getitem__(self, index):

        LS_file = os.listdir(self.input_path)[index]
        subjectname = LS_file[0:6]
        
        f_LS = os.path.join(self.input_path, LS_file)

        LS_tot = nib.load(f_LS).get_fdata()
        
        if LS_tot.shape[0] > 128 and LS_tot.shape[0] == LS_tot.shape[1]:
            excess = (LS_tot.shape[0]-128)//2
            LS_tot = LS_tot[excess:-excess, excess:-excess,:]

        return t.from_numpy(LS_tot.copy()), subjectname

#################################################################################################################################################

class LS_dataset_VolperSubjSave(Dataset):
    """This Dataset class fetches one volume at a time from the LS data loaded by LS_dataset_subjSave. The class is used when reconstructing images 
    from test set subjects. The reconstructed data is to be saved in NIfTI-files for further processing. 

    Parameters:
    - input_vols (tensor):  Complete Looping Star image data for all subjects in the computer memoy

    Returns:
    - Tensor: Torch tensor with one LS volume
    
   """
    def __init__(self, input_vols):
        self.input_vols = input_vols

        inp_shp = input_vols.shape
        self.num_vols = inp_shp[4]

    def __len__(self):
        return self.num_vols

    def __getitem__(self, index):

        LS_vol = self.input_vols[0,:,:,:,index]

        return t.tensor(np.expand_dims(LS_vol, 0))

#################################################################################################################################################

class LS_dataset_SliceperSubjSave(Dataset):
    """This Dataset class fetches one slice at a time from the LS data loaded by LS_dataset_subjSave. The class is used when reconstructing images 
    from test set subjects. The reconstructed data is to be saved in NIfTI-files for further processing. 

    Parameters:
    - input_vols (tensor):  Complete Looping Star image data for all subjects in the computer memoy

    Returns:
    - Tensor: Torch tensor with one LS slice
    
   """
    def __init__(self, input_vols):
        self.input_vols = input_vols

        inp_shp = input_vols.shape
        self.num_volumes = inp_shp[4]
        self.num_slices = inp_shp[3]

    def __len__(self):
        return self.num_volumes*self.num_slices

    def get_vol_slice_indx(self, index):
        vol_idx = index // self.num_slices
        slice_idx = (index - vol_idx*self.num_slices) % self.num_slices

        return vol_idx, slice_idx

    def __getitem__(self, index):
        vol_idx, slice_idx = self.get_vol_slice_indx(index)

        LS_slice = self.input_vols[0,:,:,slice_idx, vol_idx]

        return t.tensor(np.expand_dims(LS_slice, 0))
    


