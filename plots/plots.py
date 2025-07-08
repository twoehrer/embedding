#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from matplotlib.colors import to_rgb
import imageio

from matplotlib.colors import LinearSegmentedColormap
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_EVmod_of_weightmatrix(model, log_scale=True, title='', ax=None):
    """
    For each Linear layer in the model, compute the eigenvalues of the weight matrix,
    take their modulus, and plot them by layer index (x-axis) vs. modulus (y-axis).
    """
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

    all_moduli = []
    max_val = float('-inf')
    min_val = float('inf')

    for i, layer in enumerate(linear_layers):
        W = layer.weight.detach().cpu()
        try:
            eigvals = torch.eig(W, eigenvectors=False)[0]
            # print('layer', i, 'eigenvals',eigvals)
            
            moduli = torch.norm(eigvals, dim = 1).numpy()
            all_moduli.append((i, moduli))
            max_val = max(max_val, moduli.max())
            min_val = min(min_val, moduli.min())
        except RuntimeError:
            print(f"⚠️ torch.eig failed on Layer {i+1}. Probably not a square matrix. Set to 0")
            all_moduli.append((i, []))

    # Plot each modulus as a separate point per layer
    if ax is None:
        fig, ax = plt.subplots()
    for layer_idx, mods in all_moduli:
        for val in mods:
            print(f"Layer {layer_idx}, val: {val}")
            ax.scatter(layer_idx, val, color='purple', alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Modulus of EVs")
    if log_scale:
        ax.set_yscale("log")

    if min_val > 0 and max_val > 0:
        ax.set_ylim([min_val * 0.8, max_val * 1.2])
    else:
        ax.set_ylim([1e-4, 10])

    ax.set_xticks(list(range(len(linear_layers))))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if ax is None:
        plt.tight_layout()
        plt.show()



def plot_singular_values_of_weightmatrix(model, log_scale = True, title='', ax = None):
    """
    For each Linear layer in the model, compute the singular values using torch.svd,
    and plot them by layer index (x-axis) vs. singular value (y-axis).
    This version ensures all singular values are visible and x-ticks align with integer layer indices.
    """
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

    all_singular_values = []
    max_sv = float('-inf')
    min_sv = float('inf')

    for i, layer in enumerate(linear_layers):
        W = layer.weight.detach().cpu()
        try:
            _, S, _ = torch.svd(W)
            sv_numpy = S.numpy()
            all_singular_values.append((i, sv_numpy))
            max_sv = max(max_sv, sv_numpy.max())
            min_sv = min(min_sv, sv_numpy.min())
        except RuntimeError:
            print(f"⚠️ torch.svd failed on Layer {i+1} — possibly due to singularity.")
            all_singular_values.append((i, []))

    # Plot each singular value as a separate point per layer
    if ax is None:
        fig, ax = plt.subplots()
    for layer_idx, svals in all_singular_values:
        for sv in svals:
            ax.scatter(layer_idx, sv, color='blue', alpha=0.4)

    ax.set_title("SVs: " + title)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Singular Value (log scale)")
    if log_scale:
        ax.set_yscale("log")

    # Ensure y-axis covers all singular values
    y_max = max_sv * 1.2
    ax.set_ylim([1e-4, y_max])

    # Use integer x-ticks only for layer indices
    layer_indices = list(range(len(linear_layers)))
    ax.set_xticks(layer_indices)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
 
 
 


def psi_manual(x, func, output_type = 'sv'):
    """
    x: a tensor of shape (2,) representing a point in R^2.
    model: a function mapping R^2 to R^output_dim.
    
    Returns:
      The smallest singular value of the Jacobian of model at x.
    """
    # Ensure x is a leaf variable with gradient tracking enabled.
    x = x.clone().detach().requires_grad_(True)  
    # print(f"x shape: {x.shape}")  # Debugging line to check the shape of x
    
    
    # Compute the Jacobian using torch.autograd.functional.jacobian (compatible with Python 3.8)
    jacobian = torch.autograd.functional.jacobian(func, x, create_graph=True)
    # print(f"Jacobian shape: {jacobian.shape}")  # Debugging line to check the shape of the Jacobian
    
    # print(f"Jacobian shape after squeeze: {jacobian.shape}")  # Debugging line to check the shape after squeeze
    # Compute singular values using svdvals (available in PyTorch 1.8, compatible with Python 3.8)
    if output_type == 'sv':
        output = torch.svd(jacobian, compute_uv=False)[1] #svd interprets here the jacobian as a SQUARE matrix of the largest dimension, hence it 
    elif output_type == 'eigmods':  # Ensure jacobian is square for eigenvalue computation
        if not jacobian.shape[0] == jacobian.shape[1]: 
            output = torch.zeros_like(x)  # If not square, return zero vector
        else:
            eigs = torch.eig(jacobian, eigenvectors=False)[0]
            moduli = torch.norm(eigs, dim = 1)
            sorted_indices = torch.argsort(moduli, descending=True ) #descending order to match singular values behavior
            output = moduli[sorted_indices]
    else:
        raise ValueError("output_type must be either 'sv' or 'eigmods'")
             
    return output.detach().numpy()
  

def model_to_func(model,from_layer=0, to_layer=-1):
  
  if from_layer == 0 and to_layer == -1: # this is the case for input to last hidden layer (without output layer)
    func = lambda inp: model(inp.unsqueeze(0), output_layer = False).squeeze(0)  # Add artificial batch dimension which is needed because of batch normalization layer BatchNorm1d and remove it again from the model output.
  else: 
    func = lambda inp: model.sub_model(inp.unsqueeze(0), from_layer=from_layer, to_layer = to_layer).squeeze(0)
  
  return func
  
'''
output_type: 'sv' for singular values, 'eigmods' for eigenvalue moduli
'''
def sv_plot(func, v_index = 0, x_range = [-1,1], y_range = [-1,1], grid_size = 100, ax = None, title = '', output_type = 'sv'):
  x_values = np.linspace(x_range[0], x_range[1], grid_size)
  y_values = np.linspace(y_range[0], y_range[1], grid_size)
  psi_values = np.zeros((grid_size, grid_size, 2))
  
  # Evaluate psi(x) over the grid.
  for i, xv in enumerate(x_values):
      for j, yv in enumerate(y_values):
          # Create a 2D point as a torch tensor.
          x_point = torch.tensor([xv, yv], dtype=torch.float32)
          psi_values[j, i,:] = psi_manual(x_point, func, output_type = output_type) #one subtlety here: if there is only one SV it gets broadcast to all dimensions of psi_values[j,i,:] in the last dimension. this reduces if statements for e.g. the last layer, but we need to notice that the SINGLE SV gets plotted twice  
   

  # Here we plot the contour at a small level, e.g., 0.01.
  # CS = plt.contour(x_range, y_range, psi_values, levels=[0,0.05,0.1,0.2,0.3], colors='red')

  # Define the number of levels for the contour plot
  vmin1, vmax1 = psi_values[:, :, v_index].min(), psi_values[:, :, v_index].max()
  num_levels = 200

  levels = np.linspace(0, vmax1, num_levels)
  
  # Plot on the provided axis
  if ax is not None:
      cs = ax.contourf(x_values, y_values, psi_values[:, :, v_index], levels=levels, cmap='viridis')
      ax.set_title(title)
      ax.set_xlabel('x1')
      ax.set_ylabel('x2')
      ax.set_aspect('equal')
      return cs
  else:
    # Create the contour plot using the 'binary' colormap
    plt.figure(figsize=(8, 6))
    
    CS = plt.contourf(x_values, y_values, psi_values[:,:,v_index], levels=levels, cmap = 'viridis')
    cbar = plt.colorbar(CS)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


#this is the new function more distilled from the old one

def plot_level_sets(model, title="Prediction Level Sets", amount_levels=50, x_min = -1, x_max = 1, y_min = -1, y_max = 1, ax=None, show=True, file_name = None, footnote = None):
    """
    Plots just the contour lines of the model's predictions
    """

    
    model.eval()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid_tensor).numpy().reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    levels = np.linspace(0., 1., amount_levels).tolist()
    contour = ax.contour(xx, yy, preds, 
                          levels=levels, 
                          colors = 'k', linewidths = 0.3, alpha=1)
    
    
    
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.axis('tight')
    ax.grid(False)

    colorbar_ticks = np.linspace(0, 1, 9)
    cb = plt.colorbar(contour, ax=ax, label='Prediction Probability', ticks=colorbar_ticks)
    cb.set_ticklabels([f"{tick:.2f}" for tick in colorbar_ticks])
    
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=8)
    
    if file_name is not None:
        file_name = file_name + '.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=300, facecolor = 'white')
        print(f"Plot saved to {file_name}")
    if show and ax is None:
        plt.show()

def plot_decision_boundary(model, X, y, title="Prediction Level Sets", amount_levels=50, margin=0.2, ax=None, show=True, colorbar = True, file_name = None, show_points = True, footnote = None):
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    colors = [to_rgb("C0"), [1, 1, 1], to_rgb("C1")]
    cm = LinearSegmentedColormap.from_list("Custom", colors, N=amount_levels)

    model.eval()
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid_tensor).numpy().reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    levels = np.linspace(0., 1., amount_levels).tolist()
    contour = ax.contourf(xx, yy, preds, 
                          levels=levels, 
                          cmap=cm, alpha=0.8)
    
    
    
    if show_points == True:
        scatter = ax.scatter(X[:, 0], X[:, 1], s=25, c=y.squeeze(), cmap=cm, edgecolors='black', linewidths=0.5, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
 
    ax.axis('tight')
    ax.grid(False)
    if colorbar:
        colorbar_ticks = np.linspace(0, 1, 9)
        cb = plt.colorbar(contour, ax=ax, label='Prediction Probability', ticks=colorbar_ticks)
        cb.set_ticklabels([f"{tick:.2f}" for tick in colorbar_ticks])
    
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=8)
    
    if file_name is not None:
        file_name = file_name + '.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=300, facecolor = 'white')
        print(f"Plot saved to {file_name}")
    if show and ax is None:
        plt.show()
        


@torch.no_grad()
def visualize_classification(model, data, label, grad = None, fig_name=None, footnote=None, contour = True, x1lims = [-2, 2], x2lims = [-2, 2]):
    
    
    x1lower, x1upper = x1lims
    x2lower, x2upper = x2lims

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0", zorder = 1)
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1", zorder = 1)

    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)
    # plt.legend()
    if not grad == None:
        for i in range(len(data[:, 0])):
            plt.arrow(data[i, 0], data[i, 1], grad[i, 0], grad[i, 1],
                    head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=0.5, length_includes_head = True)

   
    model.to(device)
    # creates the RGB values of the two scatter plot colors.
    # c0 = torch.Tensor(to_rgba("C0")).to(device)
    # c1 = torch.Tensor(to_rgba("C1")).to(device)

    

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds, _ = model(model_inputs)
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    # now we only want to have the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    # Specifying "None" in a dimension creates a new one. The rgb values hence get rescaled according to the prediction
    # output_image = (1 - preds) * c1[None, None] + preds * c0[None, None]
    # # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    # output_image = output_image.cpu().numpy()
    # plt.imshow(output_image, origin='lower', extent=(x1lower, x1upper, x2lower, x2upper), zorder = -1)
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])
    # plt.axis('scaled')

    # labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.numpy()]
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")]
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,8).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')



    # preds_contour = preds.view(len(x1), len(x1)).detach()
    # plt.contourf(xx1, xx2, preds_contour, alpha=1)
    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
    return fig


@torch.no_grad()
def classification_levelsets(model, fig_name=None, footnote=None, contour = True, amount_levels = 8, plotlim = [-2, 2]):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(5, 5), dpi=100)
    
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)

    
   
    model.to(device)

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    
    preds, _ = model(model_inputs)
    
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    #we only need the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])

    ax = plt.gca()
    ax.set_aspect('equal') 
    
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,amount_levels).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')
    

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
    # else: plt.show()
    else:
        return fig, ax
    
def loss_evolution(trainer, epoch, filename = '', figsize = None, footnote = None):
    print(f'{epoch = }')
    fig = plt.figure(dpi = 100, figsize=(figsize))
    labelsize = 10

    #plot whole loss history in semi-transparent
    epoch_scale = range(1,len(trainer.histories['epoch_loss_history']) + 1)
    epoch_scale = list(epoch_scale)
    plt.plot(epoch_scale,trainer.histories['epoch_loss_history'], 'k', alpha = 0.5 )
    plt.plot(epoch_scale, trainer.histories['epoch_loss_rob_history'], 'C2--', zorder = -1, alpha = 0.5)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        standard_loss_term = [loss - rob for loss, rob in zip(trainer.histories['epoch_loss_history'],trainer.histories['epoch_loss_rob_history'])]
        plt.plot(epoch_scale, standard_loss_term,'C1--', alpha = 0.5)
        leg = plt.legend(['total loss', 'gradient term', 'standard term'], prop= {'size': labelsize})
    else: leg = plt.legend(['standard loss', '(inactive) gradient term'], prop= {'size': labelsize})
        
    #set alpha to 1
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.plot(epoch_scale[0:epoch], trainer.histories['epoch_loss_history'][0:epoch], color = 'k')
    plt.scatter(epoch, trainer.histories['epoch_loss_history'][epoch-1], color = 'k' , zorder = 1)
    
    plt.plot(epoch_scale[0:epoch], trainer.histories['epoch_loss_rob_history'][0:epoch], 'C2--')
    plt.scatter(epoch, trainer.histories['epoch_loss_rob_history'][epoch - 1], color = 'C2', zorder = 1)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        plt.plot(epoch_scale[0:epoch], standard_loss_term[0:epoch],'--', color = 'C1')
        plt.scatter(epoch, standard_loss_term[epoch - 1], color = 'C1', zorder = 1)
        
    plt.xlim(1, len(trainer.histories['epoch_loss_history']))
    # plt.ylim([0,0.75])
    plt.yticks(np.arange(0,1,0.25))
    plt.grid(zorder = -2)
    # plt.tight_layout()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.set_aspect('auto')
    ax.set_axisbelow(True)
    plt.xlabel('Epochs', size = labelsize)
    if trainer.eps > 0:
        plt.ylabel('Loss Robust', size = labelsize)
        
    else:
        plt.ylabel('Loss Standard', size = labelsize)

    if footnote:
        plt.figtext(0.5, -0.005, footnote, ha="center", fontsize=9)

    if not filename == '':
        plt.savefig(filename + '.png', bbox_inches='tight', dpi=100, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
        
    else:
        plt.show()
        print('no filename given')
        

def comparison_plot(filename1, title1, filename2, title2, filename_output, figsize = None, show = False, dpi = 100):
    plt.figure(dpi = dpi, figsize=figsize)
    plt.subplot(121)
    sub1 = imageio.imread(filename1)
    plt.imshow(sub1)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(122)
    sub2 = imageio.imread(filename2)
    plt.imshow(sub2)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(filename_output, bbox_inches='tight', dpi=dpi, format='png', facecolor = 'white')
    if show: plt.show()
    else:
        plt.gca()
        plt.close()
        
        
def train_to_classifier_imgs(model, trainer, dataloader, subfolder, num_epochs, plotfreq, filename = '', plotlim = [-2, 2]):
    
    if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    fig_name_base = os.path.join(subfolder,'') #os independent file path

    for epoch in range(0,num_epochs,plotfreq):
        trainer.train(dataloader, plotfreq)
        epoch_trained = epoch + plotfreq
        classification_levelsets(model, fig_name = fig_name_base + filename + str(epoch_trained), footnote = f'epoch = {epoch_trained}', plotlim = plotlim)
        print(f'\n Plot {epoch_trained =}')