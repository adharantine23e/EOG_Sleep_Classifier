import mne
import os
import warnings
import wandb
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import pytorch_warmup as warmup
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from utils import augmentation_method, filter_data
from model import EOGClassifier

class EOGRawProcessor:
    def __init__(self, data_dir = r"Input your dir here"):
        self.data_dir = data_dir
        self.data = None
        self.labels = None
        
    def degrade_value(self, value: List[int], num_factor: int = 5) -> np.ndarray:
        arr = np.array(value) - 1
        match num_factor:
            case 5: # Wake, N1, N2, N3, REM
                return arr
            case 4: # Wake, Light-Sleep, Deep-Sleep, REM
                arr = np.where(arr == 2, 1, arr)  # N2 -> Light-Sleep
                return np.where(arr > 2, arr - 1, arr)
            case 3: # Wake, Sleep, REM
                arr = np.where((arr >= 1) & (arr <= 3), 1, arr)  # N1,N2,N3 -> Sleep
                return np.where(arr > 3, 2, arr)
            case 2: # Wake, Sleep
                return np.where(arr >= 1, 1, arr)
            case _:
                return ValueError("Invalid num_factor value Must be 2 -> 5")
            
    def load_data(self)-> Tuple[List[str], List[str]]:
        hyp_files = []
        psg_files = [f for f in os.listdir(self.data_dir) if f.endswith('-PSG.edf')]

        for i, directory in enumerate(psg_files):
            psg_name = directory.split('-')[0]
            prefix = psg_name[:-2]
            hyp_files = [f for f in os.listdir(self.data_dir) if f.startswith(prefix) and f.endswith('-Hypnogram.edf')]
            if hyp_file:
                hyp_file = hyp_file[0]
                print("PSG File:", psg_name)
                print("Hypnogram File:", hyp_file)
                # Perform further processing or operations with the corresponding files
        else:
                print("No corresponding Hypnogram file found for PSG file:", psg_name)
        hyp_files.append(hyp_file)
        return psg_files, hyp_files
    
    def create_Epochs_data(self) -> mne.Epochs:
        raw_list = []
        psg_files, hyp_files = self.load_data()

        for f in psg_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw =  mne.io.read_raw_edf(os.path.join(self.data_dir, f), preload=False, verbose=0, stim_channel='Event marker',
                                            infer_types=True, misc=['rectal'])
                
                # Proceed with the coresponding hyp file
                psg_names = os.path.basename(raw.filenames[0]).split("-")[0]
                prefix = psg_names[:-2]
                for j, hyp_file in enumerate(hyp_files):
                    if hyp_file.startswith(prefix):
                        print(f"The Hypnogram file that being processed: {hyp_file}")
                        Annotation = mne.read_annotations(os.path.join(self.data_dir, hyp_file))
                        # Crop the first and last 30 minutes of the record
                        Annotation.crop(Annotation[1]['onset'] - 30 * 60, Annotation[-2]['onset'] + 30 * 60)
                    else:
                        print("Can not read the file: ", hyp_file)
            raw.set_annotations(Annotation)
            raw_list.append(raw)
        
        # Create a concatenate raw object
        raw_concat = mne.concatenate_raws(raw_list)
        # Create Epoch object
        annotation_desc_event_id = {'Sleep stage W': 0,
                                    'Sleep stage 1': 1,
                                    'Sleep stage 2': 2,
                                    'Sleep stage 3': 3,
                                    'Sleep stage 4': 3,
                                    'Sleep stage R': 4}
        # Join stage 3 and 4 together
        event_id = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3/4': 3,
            'Sleep stage R': 4
        }
        event, annot = mne.events_from_annotations(raw_concat, event_id=annotation_desc_event_id, chunk_duration= 30.0)
        tmax = 30.0 - 1. /raw_concat.info['sfreq']
        Epoch_data = mne.Epochs(raw=raw_concat, events=event,
                                event_id=event_id,
                                tmin=0, tmax=tmax,
                                baseline=None,
                                preload= False)
        return Epoch_data


    def main(self, filters: bool = True, augmentation: bool = True, normalizer: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Epoch_data = self.create_Epochs_data()  
        # Extract the horizontal channel and the events
        eog_raw_data = Epoch_data.get_data(picks = "horizontal")
        eog_raw_data = eog_raw_data.reshape((len(eog_raw_data), -1))

        labels = Epoch_data.events[:, 2]
        labels = self.degrade_value(value = labels, num_factor = 5)

        # Filters
        if filters:
            eog_raw_data = filter_data(eog_raw_data, sfreq= 100.0, l_freq= 0.3, h_freq= 15.0)
        
        # Split the data to train, val, test
        train_data, test_data, train_labels, test_labels = train_test_split(eog_raw_data, labels, test_size=0.15, random_state=42, shuffle=True)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.15, random_state=42, shuffle= True)

        # Augmentation
        if augmentation:
            eog_raw_data, labels = augmentation_method(data =train_data, labels = train_labels)
        
        # Normalizer 
        if normalizer:
            Nmax = preprocessing.RobustScaler()
            train_data = Nmax.fit_transform(train_data)
            val_data = Nmax.transform(val_data)
            test_data = Nmax.transform(test_data)

        train_data = np.expand_dims(train_data, axis = 1)
        val_data = np.expand_dims(val_data, axis = 1)
        test_data= np.expand_dims(test_data, axis = 1)

        return train_data, val_data, test_data, train_labels, val_labels, test_labels

class EOGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.labels[idx]).long()
        return sample, label
    
class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 device: torch.device, epoch: int, lr: float = 1e-4):
        self.gradient_flow = []
        self.iters = len(train_loader) * epoch
        self.classes = ('0', '1', '2', '3', '4')
        self.regulizer = "L2"
        self.lambda_req = 0.01
        self.pruning = False
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.save_model = SaveBestModel()
        self.optimizer = optim.Adam(self.model.parameters(),lr= self.lr)
        self.criterion = nn.CrossEntropyLoss()
        # Define LR scheduler and warmup scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                 T_max= self.iters,
                                                                 eta_min= 5e-6)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        wandb.watch(self.model, self.criterion, log = "all")
        progress_bar  = tqdm(self.train_loader, desc= "Training")
        for idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss_score = self.criterion(output, target)
            # Regulizer
            if self.regulizer == "L2":
                L2_norm = sum(param.pow(2).sum() for param in self.model.parameters())
                loss_score += self.lambda_req * L2_norm
            elif self.regulizer == "L1":
                L1_norm = sum(param.abs().sum() for param in self.model.parameters())
                loss_score += self.lambda_req * L1_norm
            train_loss += loss_score.item()

            # Calculate accuracy
            _, prediction = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (prediction == target).sum().item()

            #  Backward pass
            loss_score.backward()
            self.optimizer.step()
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step()
                
            # Prunning if necessary (Prune for every 10 epochs for  only Conv1d and Linear layers)
            if self.pruning:
                if (epoch + 1) % 10 == 0:
                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                            prune.l1_unstructured(module, name = "weight", amount= 0.1)
                            prune.remove(module= module, name = "weight")
            
        # Log back to wandb and progress bar
        current_lr = self.optimizer.param_groups[0]['lr']
        train_accuracy = round((train_correct / train_total)*100, 4)
        wandb.log({'Train Loss': loss_score.item(), "Train Accuracy": train_accuracy, "Learning Rate": current_lr})
        progress_bar.set_postfix({ 
            "Loss": f"{loss_score.item():.4f}",
            "Accuracy": f"{train_accuracy}"})

        return train_loss / len(self.train_loader), train_accuracy 
    

    def val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc = "Validation")
            for idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss_score = self.criterion(output, target)
                val_loss += loss_score.item()
                _, prediction = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (prediction == target).sum().item()
            # Log back to wandb and progress bar
            val_accuracy =  round((val_correct / val_total)* 100, 4)
            wandb.log({
                'Validation Loss': val_loss/val_total,
                'Validation Overall Loss': val_loss/len(test_loader), 
                'Validation Accuracy': val_accuracy
            })

            progress_bar.set_postfix({
                "Loss": f"{loss_score.item():.4f}",
                "Accuracy": f"{val_accuracy}"
            })

        return val_loss / len(self.val_loader), val_accuracy
    
    def train(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)

            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()

            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            # Save the best model
            self.save_model(val_accuracy, epoch,
                            self.model, self.optimizer, self.criterion)
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 1)  
        plt.tight_layout()
        plt.show()

class SaveBestModel():
    def __init__(self, best_valid_accuracy=float('-inf')):
        self.best_valid_accuracy = best_valid_accuracy

    def __call__(self, current_valid_accuracy, epoch, model, optimizer, criterion):
        if current_valid_accuracy > self.best_valid_accuracy:
            self.best_valid_accuracy = current_valid_accuracy
            print(f"\nBest validation accuracy: {self.best_valid_accuracy}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, 'app/best_model.pth')



if __name__ == "__main__":
    num_epoch = 50
    wandb.login()
    Raw_processor = EOGRawProcessor()
    train_data, val_data, test_data, train_labels, val_labels, test_labels = Raw_processor.main()

    train_set = EOGDataset(train_data, train_labels)
    val_set = EOGDataset(val_data, val_labels)
    test_set = EOGDataset(test_data, test_labels)

    train_loader = DataLoader(train_set, batch_size = 100, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = 100, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = 100, shuffle = False)

    model = EOGClassifier(num_class = 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model = model,
        train_loader = train_loader,
        val_loader= val_loader,
        device= device,
        epoch= num_epoch,
        lr= 1e-4
    )
    train_losses, val_losses, train_accuracies, val_accuracies = trainer.train(
        num_epochs= num_epoch
    )
    trainer.plot_training_history()
  
