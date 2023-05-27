import sys
sys.path.append('C:/Users/MariusER/Documents/GitHub/Master-Project-LoopingStar/DL_training_py/3D-Unet')
import torch as t
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from dataset_management import LS_dataset_SliceperSubj, LS_dataset_SliceperSubjSave

from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import os


def train_loop_tqdm(epoch, train_loader, model, optimizer, loss_fn, device, num_augment = 0, num_vol=None):
    ''' Training loop. This function use PyTorch utils to calculate the loss and perform backpropagation.

    Parameters:
    - epochs (int):                 Current training epoch  
    - train_loader (Dataloader):    Dataloader that loads training data in batches
    - model (nn.Module):            The trainable deep neural network
    - optimizer (t.optim):          Model parametr optimizer
    - loss_fn:                      Loss function
    - device:                       t.device("cuda") or t.device("cpu")
    - num_augment (int):            Number of augmentation procedures to include
    - num_vol (int):                Number of volumes per subject to include
 
    Returns:
    - loss_train (float):           Loss for this training epoch
    '''
    model.train()
    t.cuda.empty_cache()
    loss_train = 0.0
    with tqdm(train_loader, unit='batch', bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}') as t_train_loader:
        for inputs, labels in t_train_loader:
            t_train_loader.set_description(f'Training  Epoch  {epoch}')

            subj_set_train = LS_dataset_SliceperSubj(inputs, labels, num_vol=num_vol, num_augment=num_augment)
            subj_loader_train = DataLoader(subj_set_train, batch_size=90, shuffle=True)
            for input, label in subj_loader_train:
                input = input.to(device=device, dtype=t.float32) 

                output = model(input)
                del input
                label = label.to(device=device, dtype=t.float32)

                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_train += loss.item()
                t_train_loader.set_postfix(loss=loss_train)

    return loss_train


def val_loop_tqdm(epoch, val_loader, model, loss_fn, device, num_augment = 0, num_vol=None, lr_decay = None):
    ''' Validation loop. This function calculate th loss in the validation set.

    Parameters:
    - epochs (int):                 Current training epoch  
    - val_loader (Dataloader):      Dataloader that loads validation data in batches
    - model (nn.Module):            The trainable deep neural network
    - loss_fn:                      Loss function
    - device:                       t.device("cuda") or t.device("cpu")
    - num_augment (int):            Number of augmentation procedures to include
    - num_vol (int):                Number of volumes per subject to include
    - lr_decay:                     Learning rate scheduler (not used)
 
    Returns:
    - loss_val (float):             Loss for this validation epoch
    - hist_out:                     100-bin histogram counts of image values between 0 and 1 (not used)
    '''
    with t.no_grad():
        model.eval()
        t.cuda.empty_cache()
        loss_val = 0.0
        hist_out = t.zeros(100)
        with tqdm(val_loader, unit='batch', bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}') as t_val_loader:
            for val_inputs, val_labels in t_val_loader:
                t_val_loader.set_description(f'Validating Epoch {epoch}')

                subj_set_val = LS_dataset_SliceperSubj(val_inputs, val_labels, num_vol=num_vol, num_augment=num_augment)
                subj_loader_val = DataLoader(subj_set_val, batch_size=90, shuffle=False)

                for val_input, val_label in subj_loader_val:
                    val_input = val_input.to(device=device, dtype=t.float32)

                    val_output = model(val_input)
                    del val_input
                    val_label = val_label.to(device=device, dtype=t.float32)
            
                    val_loss = loss_fn(val_output, val_label)

                    val_output = val_output.to(device="cpu")
                    hist_out += t.histc(val_output, bins=100, max=1.05, min=-0.05)
            
                    loss_val += val_loss.item()
                    t_val_loader.set_postfix(loss=loss_val)

    if not lr_decay == None:
        lr_decay.step(loss_val)

    return loss_val, hist_out.numpy()


def train_and_eval(epochs, train_loader, val_loader, model, optimizer, loss_fn, device, num_augment = 0, model_savename = None):
    ''' Trainer function that perform complete model training on the training data and intermediate evaluation on the validation data.

    Parameters:
    - epochs (int):                 Number of training epochs  
    - train_loader (Dataloader):    Dataloader that loads training data in batches
    - val_loader (Dataloader):      Dataloader that loads validation data in batches
    - model (nn.Module):            The trainable deep neural network
    - optimizer (t.optim):          Model parametr optimizer
    - loss_fn:                      Loss function
    - device:                       t.device("cuda") or t.device("cpu")
    - num_augment (int):            Number of augmentation procedures to include
    - model_savename (str):         Filename to save the best model as
    - num_vol (int):                Number of volumes per subject to include
 
    Returns:
    - loss_train (float):           Loss for this training epoch
    - loss_val (float):             Loss for this validation epoch
    - hist_out:                     100-bin histogram counts of image values between 0 and 1 (not used)
    '''
    losses_train = []
    losses_val = []
    val_hists = []
    optimizer.zero_grad(set_to_none=True)

    if not model_savename == None:
        if not isinstance(model_savename, str) or not model_savename[-3:] == ".pt":
            raise Exception("A savename for the model is not given or is of the wrong type")
    
    for epoch in range(1, epochs + 1):
         
        loss_train = train_loop_tqdm(epoch, train_loader, model, optimizer, loss_fn, device, num_augment=num_augment)

        loss_val, hist_val = val_loop_tqdm(epoch, val_loader, model, loss_fn, device, num_augment=num_augment)

        losses_train.append(loss_train)
        losses_val.append(loss_val)
        val_hists.append(hist_val.tolist())

        ### Model Saving ###
        if not model_savename == None:
            if(epoch == 1):
                t.save(model.state_dict(), model_savename)
                min_loss = loss_val
            elif(loss_val < min_loss):
                t.save(model.state_dict(), model_savename)
                min_loss = loss_val
            else:
                continue

    return losses_train, losses_val, val_hists

################################## Eval: ##################################

def take_a_look(model, loader, device, subindex = 0, figsize=(10,10)):
    ''' Function that pass one example volume through the network and displays the reconstruction results

    Parameters:
    - model (nn.Module):    Deep neural network
    - loader (Dataloader):  Dataloader that loads data in batches
    - device:               t.device("cuda") or t.device("cpu")
    - subindex (int):       Which subject to load   
    - figsize (tuple):      Size of the displayed figure           
 
    '''
    with t.no_grad():
        model.to(device=device, dtype=t.float32)
        model.eval()
    
        i = 0
        for inputs, labels in loader:
                if i == subindex:
                        subj_set_test = LS_dataset_SliceperSubj(inputs, labels)
                        subj_loader_test = DataLoader(subj_set_test, batch_size=90, shuffle=False)
                        for input_test, label_test in subj_loader_test:
                                input_test = input_test.to(device=device, dtype=t.float32) 

                                output_test = model(input_test)

                                break
                        break
                i += 1
        
        output_test = output_test.to('cpu')
        input_test = input_test.to('cpu')

        arr1 = t.cat((t.rot90(label_test.squeeze()[40,:,:]), t.rot90(input_test.squeeze()[40,:,:]), t.rot90(output_test.squeeze()[40,:,:])), dim=1)
        arr2 = t.cat((t.rot90(label_test.squeeze()[45,:,:]), t.rot90(input_test.squeeze()[45,:,:]), t.rot90(output_test.squeeze()[45,:,:])), dim=1)
        arr3 = t.cat((t.rot90(label_test.squeeze()[50,:,:]), t.rot90(input_test.squeeze()[50,:,:]), t.rot90(output_test.squeeze()[50,:,:])), dim=1)
        arr4 = t.cat((t.rot90(label_test.squeeze()[60,:,:]), t.rot90(input_test.squeeze()[60,:,:]), t.rot90(output_test.squeeze()[60,:,:])), dim=1)
        arr5 = t.cat((t.rot90(label_test.squeeze()[70,:,:]), t.rot90(input_test.squeeze()[70,:,:]), t.rot90(output_test.squeeze()[70,:,:])), dim=1)
        arr0 = t.cat((t.rot90(label_test.squeeze()[80,:,:]), t.rot90(input_test.squeeze()[80,:,:]), t.rot90(output_test.squeeze()[80,:,:])), dim=1)

        arr = t.cat((arr1, arr2, arr3, arr4, arr5, arr0), dim = 0)

        figure = plt.figure(figsize=figsize)
        plt.imshow(arr, cmap='gray')
        plt.axis('off')
        plt.show()

def test_set_recon_n_save(model, test_loader, device, savepath, suffix):
     ''' Function saves the reconstructed test set images in NIfTI files.

    Parameters:
    - model (nn.Module):        Deep neural network
    - test_loader (Dataloader): Dataloader that loads test data in batches
    - device:                   t.device("cuda") or t.device("cpu")
    - savepath (str):           Path to save image files at   
    - suffix (str):             String to add to the end of filename           
 
    '''
     with t.no_grad():
        model.eval()
    
        for subject, SubjectID in test_loader:
                shape = subject.shape
                DL_recon = t.empty((shape[1], shape[2], shape[3], shape[4]))
                savefilename = SubjectID[0] + suffix

                subj_set_test = LS_dataset_SliceperSubjSave(subject)
                subj_loader_test = DataLoader(subj_set_test, batch_size=90, shuffle=False)
                volnr = 0
                with tqdm(subj_loader_test, unit='volume') as t_loader:
                        for subj_vol in t_loader:
                                t_loader.set_description(f'Subject {SubjectID[0]}')

                                subj_vol = subj_vol.to(device=device, dtype=t.float32) 

                                output_test = model(subj_vol)
                                output_test = output_test.to('cpu')

                                output_test = output_test.squeeze()
                                output_test = t.permute(output_test, (1,2,0))
                                DL_recon[:,:,:,volnr] = output_test
                                volnr += 1
                                t_loader.set_postfix()

                DL_recon = DL_recon.numpy()
                img = nib.Nifti1Image(DL_recon, None)
                nib.save(img, os.path.join(savepath, savefilename)) 

                del t_loader
                del subj_loader_test
                del subj_set_test
                del subject
                del DL_recon