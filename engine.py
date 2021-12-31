import torch
import torch.nn as nn
import numpy as np
import sys

from model import model
from tqdm import tqdm
from utils.helpers import draw_seg_maps
from utils.helpers import save_model_dict
from utils.metrics import eval_metric
from utils.helpers import TensorboardWriter
model_path = "model.pth"
class Trainer:
    def __init__(self, model, train_data_loader, train_dataset, 
                 valid_data_loader, valid_dataset, classes_to_train, 
                 epochs, device, lr, resume_training=None, model_path=None):
        super(Trainer, self).__init__()

        self.train_data_loader = train_data_loader
        self.train_dataset = train_dataset
        self.valid_data_loader = valid_data_loader
        self.valid_dataset = valid_dataset
        self.model = model
        self.num_classes = len(classes_to_train)
        self.epochs = epochs
        self.device = device
        self.lr = lr
       
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        print('OPTIMIZER INITIALIZED')
        self.criterion = nn.CrossEntropyLoss() 
        print('LOSS FUNCTION INITIALIZED')

        # initialize Tensorboard `SummaryWriter()`
        self.writer = TensorboardWriter()

        print(f"NUM CLASSES: {self.num_classes}")

        if resume_training == 'yes':
            print('RESUMING TRAINING')
            # load the model checkpoint
            checkpoint = torch.load(model_path)
            self.trained_epochs = checkpoint['epoch']
            self.train_iters = checkpoint['train_iters']
            self.valid_iters = checkpoint['valid_iters']
            print(f"PREVIOUSLY TRAINED EPOCHS: {self.trained_epochs}")
            if self.trained_epochs >= self.epochs:
                print('Current epochs less than previously trained epcochs...')
                print(f"Please provide greater number of epochs than {self.trained_epochs}")
                sys.exit()
            elif self.epochs > self.trained_epochs:
                #  load model weights state_dict
                 self.model.load_state_dict(checkpoint['model_state_dict'])
                 print('TRAINED MODEL WEIGHTS LOADED...')
                 # load trained optimizer state_dict
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 print('TRAINED OPTIMIZER LOADED...')

        elif resume_training == 'no':
            self.train_iters = 0
            self.valid_iters = 0
            self.trained_epochs = 0
            print('TRAINING FROM BEGINNING')

    def get_num_epochs(self):
        return self.trained_epochs

    def fit(self):
        print('Training')
        model.train()
        train_running_loss = 0.0
        train_running_inter, train_running_union = 0, 0
        train_running_correct, train_running_label = 0, 0
        # calculate the number of batches
        num_batches = int(len(self.train_dataset)/self.train_data_loader.batch_size)
        prog_bar = tqdm(self.train_data_loader, 
                        total=num_batches)
        counter = 0 # to keep track of batch counter
        for i, data in enumerate(prog_bar):
            counter += 1
            data, target = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            outputs = outputs['out']

            ##### BATCH-WISE LOSS #####
            loss = self.criterion(outputs, target)
            train_running_loss += loss.item()
            ###########################

            ##### BATCH-WISE METRICS ####
            correct, labeled, inter, union = eval_metric(outputs, 
                                                             target, 
                                                             self.num_classes)
            # for IoU
            train_running_inter += inter
            train_running_union += union
            # for pixel accuracy
            train_running_correct += correct
            train_running_label += labeled
            #############################

            ##### BACKPROPAGATION AND PARAMETER UPDATION #####
            loss.backward()
            self.optimizer.step()
            ##################################################

            ##### TENSORBOARD LOGGING #####
            train_running_IoU = 1.0 * inter / (np.spacing(1) + union)
            train_running_mIoU = train_running_IoU.mean()
            train_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
            self.writer.tensorboard_writer(
                loss, train_running_mIoU, train_running_pixacc, self.train_iters,
                phase='train'
            )
            ###############################

            prog_bar.set_description(desc=f"Loss: {loss:.4f} | mIoU: {train_running_mIoU:.4f} | PixAcc: {train_running_pixacc:.4f}")

            self.train_iters += 1
            
        ##### PER EPOCH LOSS #####
        train_loss = train_running_loss / counter
        ##########################

        ##### PER EPOCH METRICS ######
        # IoU and mIoU
        IoU = 1.0 * train_running_inter / (np.spacing(1) + train_running_union)
        mIoU = IoU.mean()
        # pixel accuracy
        pixel_acc = 1.0 * train_running_correct / (np.spacing(1) + train_running_label)
        ##############################
        return train_loss, mIoU, pixel_acc

    def validate(self, epoch):
        print('Validating')
        model.eval()
        valid_running_loss = 0.0
        valid_running_inter, valid_running_union = 0, 0
        valid_running_correct, valid_running_label = 0, 0
        # calculate the number of batches
        num_batches = int(len(self.valid_dataset)/self.valid_data_loader.batch_size)
        with torch.no_grad():
            prog_bar = tqdm(self.valid_data_loader, 
                        total=num_batches)
            counter = 0 # to keep track of batch counter
            for i, data in enumerate(prog_bar):
                counter += 1
                data, target = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(data)
                outputs = outputs['out']
                
                # save the validation segmentation maps every...
                # ... last batch of each epoch
                if i == num_batches - 1:
                    draw_seg_maps(data, outputs, epoch, i)

                ##### BATCH-WISE LOSS #####
                loss = self.criterion(outputs, target)
                valid_running_loss += loss.item()
                ###########################

                ##### BATCH-WISE METRICS ####
                correct, labeled, inter, union = eval_metric(outputs, 
                                                                target, 
                                                                self.num_classes)
                valid_running_inter += inter
                valid_running_union += union
                # for pixel accuracy
                valid_running_correct += correct
                valid_running_label += labeled
                #############################

                ##### TENSORBOARD LOGGING #####
                valid_running_IoU = 1.0 * inter / (np.spacing(1) + union)
                valid_running_mIoU = valid_running_IoU.mean()
                valid_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
                self.writer.tensorboard_writer(
                    loss, valid_running_mIoU, valid_running_pixacc, self.valid_iters, 
                    phase='valid'
                )
                ###############################

                prog_bar.set_description(desc=f"Loss: {loss:.4f} | mIoU: {valid_running_mIoU:.4f} | PixAcc: {valid_running_pixacc:.4f}")

                self.valid_iters += 1
            
        ##### PER EPOCH LOSS #####
        valid_loss = valid_running_loss / counter
        ##########################

        ##### PER EPOCH METRICS ######
        # IoU and mIoU
        IoU = 1.0 * valid_running_inter / (np.spacing(1) + valid_running_union)
        mIoU = IoU.mean()
        # pixel accuracy
        pixel_acc = 1.0 * valid_running_correct / (np.spacing(1) + valid_running_label)
        ##############################
        return valid_loss, mIoU, pixel_acc

    def save_model(self, epochs):
        save_model_dict(self.model, epochs, 
                        self.optimizer, self.criterion, 
                        self.valid_iters, self.train_iters)