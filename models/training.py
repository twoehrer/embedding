#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch.nn as nn
import numpy as np
from numpy import mean
import torch
# from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from torch.utils import data as data
from torch.utils.data import DataLoader, TensorDataset







def compute_accuracy(y_pred, y_true):
    """
    computes accuracy of predictions against ground truth labels.
    only works for 1-dim output currently
    y_pred: float32 predictions (sigmoid outputs)
    y_true: float32 ground truth labels (0 or 1)
    """
    y_pred_binary = (y_pred >= 0.5).int()
    y_true_binary = y_true.int()
    correct = (y_pred_binary == y_true_binary).sum().item()
    total = y_true.shape[0]
    return correct / total

import copy

def train_model(model, train_loader, test_loader,
                                load_file = None, epochs=300, lr=0.01, early_stopping = True, patience=300, cross_entropy=True, seed = None):
    """
    Trains the model on the provided training data and evaluates it on the test data.
    patience is the number of epochs to wait for improvement before stopping training.
    If load_file is provided, it will load the model state from the specified file instead of training.
    Returns the trained model, best accuracy, and training losses.
    """
    if load_file is None:  # Only enter retry loop if no model is being loaded
    
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if cross_entropy:
            criterion = nn.BCELoss()
        else: criterion = nn.MSELoss()

        best_acc = 0
        patience_counter = 0
        losses = []



        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(epoch_loss / len(train_loader))
            if early_stopping:
                # Evaluate on test data
                model.eval()
                with torch.no_grad():
                    acc_summed = 0.
                    counter = 0
                    for X_test, y_test in test_loader:
                        counter += 1
                        test_preds = model(X_test)
                        acc_summed += compute_accuracy(test_preds, y_test)
                    acc = acc_summed / counter
                model.train()

                if acc > best_acc:
                    best_acc = acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"⏹️ Early stopping at epoch {epoch}, best acc: {best_acc:.3f}")
                        break

            # At end, load the best model
        if patience_counter > 0:
            model.load_state_dict(best_model_state)
            
        # --- Save Checkpoint ---
        checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # Good practice to save optimizer state too
        'losses': losses,
        'seed': seed,
        'epoch': epoch, # Save the last epoch number
        'input_dim': model.input_dim, # Save hyperparameters for verification/reproducibility
        'hidden_dim': model.hidden_dim,
        'output_dim': model.output_dim,
        'num_blocks': model.num_hidden,
        'cross_entropy': cross_entropy,
        'accuracy': best_acc,
        'activation': model.activation
        }
        save_path = f'last.pth'
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved to {save_path}')
        # We have the losses from training directly

        return model, best_acc, losses  # <--- return the best model!
    
    
    else: # If loading a model, skip training and just load the model state
        
        load_path = load_file + '.pth'
    try:
        print(f"--- Loading Checkpoint from: {load_path} ---")
        checkpoint = torch.load(load_path)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load losses and seed from the checkpoint
        losses = checkpoint.get('losses', []) # Use .get for backward compatibility if 'losses' key is missing
        loaded_seed = checkpoint.get('seed', 'Not Found') # Use .get for backward compatibility

        # Optionally load optimizer state if you plan to resume training
        # optimizer = torch.optim.Adam(model.parameters()) # Re-initialize optimizer
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1 # To resume training

        # Load other saved info (optional, but good for verification)
        loaded_input_dim = checkpoint.get('input_dim', 'Not Found')
        loaded_hidden_dim = checkpoint.get('hidden_dim', 'Not Found')
        loaded_output_dim = checkpoint.get('output_dim', 'Not Found')
        loaded_num_blocks = checkpoint.get('num_blocks', 'Not Found')
        loaded_cross_entropy = checkpoint.get('cross_entropy', 'Not Found')
        last_epoch = checkpoint.get('epoch', 'Not Found')
        best_acc = checkpoint.get('accuracy', 0.0)  # Load the best accuracy
        activation = checkpoint.get('activation', 'Not Found')
    


        print(f"Model state loaded successfully.")
        print(f"Loaded training losses (Length: {len(losses)}).")
        print(f"Original training seed: {loaded_seed}")
        print(f"Model trained for {last_epoch + 1 if isinstance(last_epoch, int) else 'N/A'} epochs.")
        print(f"Saved Hyperparameters: Input={loaded_input_dim}, Hidden={loaded_hidden_dim}, Output={loaded_output_dim}, Blocks={loaded_num_blocks}, CrossEntropy={loaded_cross_entropy}")


        model.eval() # Set model to evaluation mode after loading
        print("Model set to evaluation mode.")
        
        return model, best_acc, losses

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {load_path}")
        losses = [] # Ensure losses is an empty list if loading failed


def train_until_threshold(model_class, train_loader, test_loader, 
                          load_file = None, cross_entropy=True,max_retries=10, threshold=0.95, seed = None, **model_kwargs):
    if load_file is None:
        for attempt in range(1, max_retries + 1):
            seed = np.random.randint(1000)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = model_class(**model_kwargs)
            model, acc, losses = train_model(model, train_loader, test_loader, cross_entropy=cross_entropy)
            print(f"[Attempt {attempt}] Accuracy: {acc:.3f}")
            if acc >= threshold:
                print(f"✅ Success after {attempt} attempt(s)!")
                return model, acc, losses
        print("❌ Failed to reach threshold.")
        return model, acc, losses
    else:
        print("Loading model, skipping training.")
        model = model_class(**model_kwargs)
        model, acc, losses = train_model(model, train_loader, test_loader, load_file=load_file, cross_entropy=cross_entropy, seed = seed)
        return model, acc, losses


def plot_loss_curve(losses, title="Training Loss", filename = None):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()




########################

losses = {'mse': nn.MSELoss(), 
          'cross_entropy': nn.CrossEntropyLoss(), 
          'ell1': nn.SmoothL1Loss()
}

class Trainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            #self.loss_func = losses['mse']
            self.loss_func = nn.MultiMarginLoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'acc_history': [],
                          'epoch_loss_history': [], 'epoch_acc_history': []}
        self.buffer = {'loss': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_acc = 0.
        for i, (x_batch, y_batch) in enumerate(data_loader):
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

            if not self.turnpike:                                       ## Classical empirical risk minimization
                loss = self.loss_func(y_pred, y_batch)
            else:                                                       ## Augmented empirical risk minimization
                if self.threshold>0: # l1 controls
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    ## lambda = 5*1e-3 for spheres+inside
                    loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                    for k in range(time_steps-1)]) + 0.005*l1_regularization #this was 0.005
                    
                else: #l2 controls
                    if self.fixed_projector: #maybe not needed
                        xd = torch.tensor([[6.0/0.8156, 0.5/(2*0.4525)] if x==1 else [-6.0/0.8156, -2.0/(2*0.4525)] for x in y_batch])
                        loss = self.loss_func(y_pred, y_batch.float())+sum([self.loss_func(traj[k], xd)
                                            +self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
                    else:
                        ## beta=1.5 for point clouds, trapizoidal rule to integrate
                        beta = 1.75                      
                        loss = beta*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                        for k in range(time_steps-1)])
            loss.backward()
            self.optimizer.step()
                        
            if self.cross_entropy:
                epoch_loss += self.loss_func(traj[-1], y_batch).item()   
                m = nn.Softmax()
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += self.loss_func(y_pred, y_batch).item()
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(self.loss_func(traj[-1], y_batch).item()))
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                       
                    else:
                        print("Loss: {:.3f}".format(self.loss_func(y_pred, y_batch).item()))
                        
            self.buffer['loss'].append(self.loss_func(traj[-1], y_batch).item())
            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if not self.fixed_projector and self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)



class doublebackTrainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    -- eps: Set a strength for the extra loss term that penalizes the gradients of the original loss
    -- The float eps_comp records the gradient of the standard loss even when robust training is not active (for comparison). Only to be used with eps = 0
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False, eps = 0.01, l2_factor = 0, eps_comp = 0., db_type = 'l1'):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            # self.loss_func = losses['mse']
            self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'loss_rob_history': [],'acc_history': [],
                          'epoch_loss_history': [], 'epoch_loss_rob_history': [],  'epoch_acc_history': []}
        self.buffer = {'loss': [], 'loss_rob': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')
        self.eps = eps
        self.eps_comp = eps_comp
        self.l2_factor = l2_factor
        self.db_type = db_type
        
        # logging_dir='runs/our_experiment'
        # writer = SummaryWriter(logging_dir)

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_loss_rob = 0.
        epoch_acc = 0.

        
        #If eps = 0, we have standard training, if eps_comp is greater 0, we have standard training but record the gradient term as comparison
        #if eps > 0 we activate robust training and record the gradient term
        eps_eff = max(self.eps_comp, self.eps)
        # print(eps_eff)
        loss_max = torch.tensor(0.)


        x_batch_grad = torch.tensor(0.).to(self.device)
        
        for i, (x_batch, y_batch) in enumerate(data_loader):
                # if i == 0:
                #     print('first data batch', x_batch[0], y_batch[0])
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            
            
            if eps_eff > 0.: #!!!!
                x_batch.requires_grad = True #i need this for calculating the gradient term
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

                                               ## Classical empirical risk minimization
            loss = self.loss_func(y_pred, y_batch)
            loss_rob = torch.tensor(0.)
            # v = torch.tensor([0,1.])
            #adding perturbed trajectories
            
            if self.l2_factor > 0:
                for param in self.model.parameters():
                    l2_regularization = param.norm()
                    loss += self.l2_factor * l2_regularization
            
            if eps_eff > 0.:
                x_batch_grad = torch.autograd.grad(loss, x_batch, create_graph=True, retain_graph=True)[0] #not sure if retrain_graph is necessary here
                
                if self.db_type == 'l1':
                    loss_rob = x_batch_grad.abs().sum() #this corresponds to linfty defense
                    
                if self.db_type == 'l2':
                    loss_rob = x_batch_grad.norm() #this corresponds to l2 defense
                    
                loss_rob = eps_eff * loss_rob
            

            

            if (self.eps > 0.) and (self.eps == eps_eff): #robust loss term is active or is logged + make sure there is no confusing between epsilon of logging and training epsilon
                loss = (1-self.eps)*loss + loss_rob
                # print(f'{loss=}')
                # loss = (1-eps) * loss + eps * adj_term #was 0.005 before
            loss.backward()
            self.optimizer.step()
        
            
            if self.cross_entropy:
                epoch_loss += loss.item()
                epoch_loss_rob += loss_rob.item() 
                m = nn.Softmax(dim = 1)
                # print(y_pred.size())
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += loss.item()
                epoch_loss_rob += loss_rob.item()
                
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(loss))
                        print("Robust Term Loss: {:.3f}".format(loss_rob))
                        
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                       
                    else:
                        print("Loss: {:.3f}".format(loss))
                        
            self.buffer['loss'].append(loss.item())
            self.buffer['loss_rob'].append(loss_rob.item())
            
            
            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                self.histories['loss_rob_history'].append(mean(self.buffer['loss_rob']))
                if not self.fixed_projector and self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['loss_rob'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        self.histories['epoch_loss_rob_history'].append(epoch_loss_rob / len(data_loader))
        
        # self.histories['ep']
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)
    
    
    
    
class epsTrainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False, eps = 0.01, alpha = 0.01):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            #self.loss_func = losses['mse']
            self.loss_func = nn.MultiMarginLoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'acc_history': [],
                          'epoch_loss_history': [], 'epoch_acc_history': []}
        self.buffer = {'loss': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')
        self.eps = eps
        self.alpha = alpha #strength of robustness term

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_acc = 0.

        v_steps = 5
        v = torch.zeros(v_steps,2)
        eps = self.eps
        alpha = self.alpha
        loss_max = torch.tensor(0.)


        
        # for k in range(v_steps):
        #     t = k*(2*torch.tensor(math.pi))/v_steps
        #     v[k] = torch.tensor([torch.sin(t),torch.cos(t)])
    #generate perturbed directions
        x_batch_grad = torch.tensor(0.)
        for i, (x_batch, y_batch) in enumerate(data_loader):
                # if i == 0:
                #     print('first data batch', x_batch[0], y_batch[0])
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            x_batch.requires_grad = True #i need this for Fast sign gradient method
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

            if not self.turnpike:                                       ## Classical empirical risk minimization
                loss = self.loss_func(y_pred, y_batch)
                # v = torch.tensor([0,1.])
                #adding perturbed trajectories
                
                if eps > 0.:
                    # loss_max = torch.tensor(0.)
                    
                    #Generate the sign gradient vector
                    # loss.backward() #previously here i had retain_graph=True. i am not sure why i thought i needed it
                    # x_batch_grad = x_batch.grad.data.sign()

                    x_batch_grad = torch.autograd.grad(loss, x_batch, create_graph=True, retain_graph=True)[0]
                    x_batch_grad = x_batch_grad.sign()
                    # print('grad size',x_batch_grad.size())
                    # print('size', x_batch.size())
                    
                    # for k in range(v_steps):
                    #     y_eps, traj_eps = self.model(x_batch + eps*v[k]) #model for perturbed input
                    #     loss_v = (traj_eps - traj).abs().sum(dim = 0) #for trapezoidal rule. endpoints not regarded atm
                    #     loss_max = torch.maximum(loss_max,loss_v)
                        # print('loss max', loss_max.sum())
                        # print('loss_v', loss_v)
                        # print('loss max',loss_max)
                    # loss += 0.005*loss_max.sum()
                    # print('loss',loss)

                   

                    
                    ###########################
                    #this should only add an extra loss to the batch items that differ from the unperturbed prediction more than pert
                    y_pred_eps, _ = self.model(x_batch + eps * x_batch_grad)
                    # y_pred, _ = self.model(x_batch)
                    
                    # pert = 0.01
                    # diff = torch.abs(y_pred_eps - y_pred)
                    # cond = diff > pert
                    # y_eff = torch.where(cond, y_pred_eps, torch.tensor(0, dtype=y_pred_eps.dtype))
                    y_eff = y_pred_eps #comment this if you use other
                    ############################

                    # print('y_eff', y_eff)
                    loss = (1-alpha) * loss + alpha * self.loss_func(y_eff, y_batch) #was 0.005 before
            else:                                                       ## Augmented empirical risk minimization
                if self.threshold>0: # l1 controls
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    ## lambda = 5*1e-3 for spheres+inside
                    loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                    for k in range(time_steps-1)]) + 0.005*l1_regularization #this was 0.005
                    
                else: #l2 controls
                    if self.fixed_projector: #maybe not needed
                        xd = torch.tensor([[6.0/0.8156, 0.5/(2*0.4525)] if x==1 else [-6.0/0.8156, -2.0/(2*0.4525)] for x in y_batch])
                        loss = self.loss_func(y_pred, y_batch.float())+sum([self.loss_func(traj[k], xd)
                                            +self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
                    else:
                        ## beta=1.5 for point clouds, trapizoidal rule to integrate
                        beta = 1.75                      
                        loss = beta*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                        for k in range(time_steps-1)])
            loss.backward()
            self.optimizer.step()
        
            if self.threshold>0: 
                self.model.apply(clipper)       # We apply the Linfty constraint to the trained parameters
            
            if self.cross_entropy:
                epoch_loss += self.loss_func(traj[-1], y_batch).item()   
                m = nn.Softmax()
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += self.loss_func(y_pred, y_batch).item()
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(self.loss_func(traj[-1], y_batch).item()))
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                       
                    else:
                        print("Loss: {:.3f}".format(self.loss_func(y_pred, y_batch).item()))
                        
            self.buffer['loss'].append(self.loss_func(traj[-1], y_batch).item())
            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if not self.fixed_projector and self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)

    def x_grad(self, x_batch, y_batch):
        x_batch.requires_grad = True

        x_batch_grad = torch.tensor(0.)
        
        y_pred, _ = self.model(x_batch)
        loss = self.loss_func(y_pred, y_batch)

        self.optimizer.zero_grad()
        
        
        x_batch_grad = torch.autograd.grad(loss, x_batch)[0]
        x_batch.requires_grad = False
        return x_batch_grad

        
                

                                             ## Classical empirical risk minimization
                

def create_dataloader(data_type, data_size = 3000, noise = 0.15, factor = 0.15, random_state = 1, shuffle = True, plotlim = [-2, 2], label = 'scalar', ticks = True, markersize = 50):
    label_types = ['scalar', 'vector']
    if label not in label_types:
        raise ValueError("Invalid label type. Expected one of: %s" % label_types)
    
    
    if data_type == 'circles':
        X, y = make_circles(data_size, noise=noise, factor=factor, random_state=random_state, shuffle = shuffle)


        
    elif data_type == 'blobs':
        centers = [[-1, -1], [1, 1]]
        X, y = make_blobs(
    n_samples=data_size, centers=centers, cluster_std=noise, random_state=random_state)
        
        
    elif data_type == 'moons':
        X, y = make_moons(data_size, noise = noise, shuffle = shuffle , random_state = random_state)
    
    
    elif data_type == 'xor':
        X = torch.randint(low=0, high=2, size=(data_size, 2), dtype=torch.float32)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).float()
        # y = y.to(torch.int64)
        X += noise * torch.randn(X.shape)
        
    elif data_type == 'circles_buffer':
        X_pre = torch.empty((data_size, 2))
        
        #uniform distribution on intervall [-2,2] but data is standard transformed later on
        X_pre[:, 0] = torch.rand(data_size) * 3 - 1.5
        X_pre[:, 1] = torch.rand(data_size) * 3 - 1.5

        #exclude data points in buffer zone
        norms = torch.norm(X_pre, dim=1)
        condition = torch.logical_or(norms < (1 - factor), norms > (1 + factor))
        X = X_pre[condition, :]

        #asign labels for inner and outer area
        norms = norms[condition]
        y = (norms < 1 - factor).float()
        
        
    else: 
        print('datatype not supported')
        return None, None
    
    if label == 'vector':
        y = np.array([(2., 0.) if label == 1 else (-2., 0.) for label in y])

    g = torch.Generator()
    g.manual_seed(random_state)
    
    if data_type != 'circles_buffer':
        X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state, shuffle = shuffle)

    X_train = torch.Tensor(X_train) # transform to torch tensor for dataloader
    y_train = torch.Tensor(y_train) #transform to torch tensor for dataloader

    X_test = torch.Tensor(X_test) # transform to torch tensor for dataloader
    y_test = torch.Tensor(y_test) #transform to torch tensor for dataloader


    if label == 'scalar':
        X_train = X_train.type(torch.float32)  #type of orginial pickle.load data
        y_train = y_train.type(torch.int64) #dtype of original picle.load data

        X_test = X_test.type(torch.float32)  #type of orginial pickle.load data
        y_test = y_test.type(torch.int64) #dtype of original picle.load data
        
        
        
    else:
        X_train = X_train.type(torch.float32)  #type of orginial pickle.load data
        y_train = y_train.type(torch.float32) #dtype of original picle.load data

        X_test = X_test.type(torch.float32)  #type of orginial pickle.load data
        y_test = y_test.type(torch.float32) #dtype of original picle.load data


    train_data = TensorDataset(X_train,y_train) # create your datset
    test_data = TensorDataset(X_test, y_test)

    train = DataLoader(train_data, batch_size=64, shuffle=shuffle, generator=g)
    test = DataLoader(test_data, batch_size=256, shuffle=shuffle, generator = g) #128 before
    if label == 'scalar':
        data_0 = X_train[y_train == 0]
        data_1 = X_train[y_train == 1]
    else:
        data_0 = X_train[y_train[:,0] > 0]
        data_1 = X_train[y_train[:,0] < 0]
    fig = plt.figure(figsize = (5,5), dpi = 100)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333",  alpha = 0.5, s = markersize)
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", alpha = 0.5, s = markersize)
    plt.xlim(plotlim)
    plt.ylim(plotlim)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel(r'$x_1$', fontsize=12)
    plt.ylabel(r'$x_2$', fontsize=12)
    if ticks == False:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('trainingset.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
    plt.show()
    
    return train, test

def make_circles_uniform(output_dim, n_samples = 2000, inner_radius = 0.5, buffer = 0.2, outer_radius = 1.0, cross_entropy = False, plot = True, batch_size = 128, filename = None, seed = None):
    """Generates a dataset of points in a ring and inside the ring.
    Args:   
        n_samples (int): Total number of samples to generate.
        inner_radius (float): Radius of the inner RING line.
        outer_radius (float): Radius of the outer RING line.
        buffer (float): Buffer between ring and circle.
        cross_entropy (bool): If True, use cross-entropy loss. If False, use MSE loss.
        output_dim (int): Dimension of the output labels. MSE allows 2d
    """
    # Generate training data
    # set random seed for reproducibility
    if seed is None:
        seed = np.random.randint(1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    print(seed)
    
    # Generate outer ring points
    n_points = n_samples // 2
    inner_radius = inner_radius #of outer ring
    outer_radius = outer_radius #of outer ring

    # Buffer between inner and outer ring
    buffer = buffer

    # Points around origin
    angles_inside = np.random.uniform(0, 2 * np.pi, n_points)
    radii_inside = (inner_radius - buffer) * np.sqrt(np.random.uniform(0, 1, n_points)) # the squareroot is to ensure uniform distribution on the disc. The larger the radius, the more points for equal distance in  x and y direction. Proof: We want a constant density in x and y direction on a disc of radius R. Then transform to polar coodinates and integrate to get the distribution function in dependence of the variable radius r as well as R.

    # Points on ring
    angles_ring = np.random.uniform(0, 2 * np.pi, n_points)
    radius_ring = np.sqrt(outer_radius ** 2 - inner_radius**2 ) * np.sqrt( np.random.uniform(0, 1, n_points) + inner_radius** 2 /(outer_radius**2 - inner_radius**2) ) 
    # radius_ring = np.random.uniform(inner_radius, outer_radius, n_points) #this needs to be modified to this one. but there is some typo at the moment
    
    x_ring = radius_ring * np.cos(angles_ring)
    y_ring = radius_ring * np.sin(angles_ring)
    ring_points = np.stack((x_ring, y_ring), axis=1)


    
    x_inside = radii_inside * np.cos(angles_inside)
    y_inside = radii_inside * np.sin(angles_inside)
    inside_points = np.stack((x_inside, y_inside), axis=1)

    # Labels
    if cross_entropy:
        labels_ring = np.ones((n_points), dtype=np.int64)
        labels_inside = np.zeros((n_points), dtype=np.int64)    
    else:
        if output_dim == 2:
            labels_ring = np.tile([1, 0], (n_points, 1))
            labels_inside = np.tile([-1, 0], (n_points, 1))
        elif output_dim == 1:
            labels_ring = np.ones((n_points, 1))
            labels_inside = -np.ones((n_points, 1))

    # Combine data
    data = np.vstack((ring_points, inside_points))
    labels = np.vstack((labels_ring, labels_inside))

    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)

    if cross_entropy:
        labels = np.concatenate((labels_ring, labels_inside))
    else:
        labels = np.vstack((labels_ring, labels_inside))
    
    
    
    # Create DataLoader
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=seed)
    
    if output_dim == 1:
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    
    if plot:
        # Plot the data
        
        if cross_entropy:
            ring_X_train = X_train[y_train == 1]
            inside_X_train = X_train[y_train == 0]
        else:
            if output_dim == 1:
                ring_X_train = X_train[y_train > 0]
                inside_X_train = X_train[y_train < 0]
            elif output_dim == 2:
                ring_X_train = X_train[y_train[:,0] > 0]
                inside_X_train = X_train[y_train[:,0] < 0]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(ring_X_train[:, 0], ring_X_train[:, 1], s=20, c='C1', alpha = 0.5, label='Ring Points')
        plt.scatter(inside_X_train[:, 0], inside_X_train[:, 1], s=20, c='C0', alpha = 0.5, label='Inside Points')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        # plt.legend()
        plt.title('Training Dataset: Ring and Inside Circle')
        plt.axis('equal')
        plt.grid(True)
        
                # Save plot if filename provided
        if filename is not None:
            plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
            print(f'Plot saved as {filename}.png')
        
        plt.show()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    if output_dim == 1:
        y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
    else:
        y_train = torch.tensor(y_train, dtype=torch.float32)  # Don't reshape for 2D outputs
        y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=n_samples, shuffle=False)
        
    return train_dataloader, test_dataloader

def make_xor(output_dim, n_samples = 2000, noise = 0.2, cross_entropy = False, plot = True, batch_size = 128, filename = None):
    """Generates xor
    """
    # Generate training data
    # set random seed for reproducibility
    seed = np.random.randint(1000)
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    data = torch.randint(low=0, high=2, size=(n_samples, 2), dtype=torch.float32)
    labels = np.logical_xor(data[:, 0] > 0, data[:, 1] > 0).float()
    data += noise * torch.randn(data.shape) - 0.5
    # Generate outer ring points
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2)
    print(X_train[:5,:])
    print(y_train[:5])
    
    if plot:
        # Plot the data
        
        data_0 = X_train[y_train[:] == 0]
        data_1 = X_train[y_train[:] == 1]
        plt.figure(figsize=(8, 8))
        plt.scatter(data_0[:, 0], data_0[:, 1], s=20, c='C1', alpha = 0.5, label='Ring Points')
        plt.scatter(data_1[:, 0], data_1[:, 1], s=20, c='C0', alpha = 0.5, label='Inside Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.legend()
        plt.title('Training Dataset: Ring and Inside Points')
        plt.axis('equal')
        plt.grid(True)
        
                # Save plot if filename provided
        if filename is not None:
            plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
            print(f'Plot saved as {filename}.png')
        
        plt.show()
    

    # Convert to tensors
    # data_tensor = torch.tensor(data, dtype=torch.float32)

    if cross_entropy:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(labels_tensor[:1])
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
    


    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    if output_dim == 1:
        y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
    else:
        y_train = torch.tensor(y_train, dtype=torch.float32)  # Don't reshape for 2D outputs
        y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=n_samples, shuffle=False)
        
    return train_dataloader, test_dataloader