from typing import Tuple, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.optim import Optimizer
import datasets_orig
import unet_orig
from tqdm import tqdm

# Darcy loss function
def darcy_loss(model, inp):
    # Takes in the k,pres,phi and outputs the prediction across the image.
    inp = inp.requires_grad_(True)
    out = model(inp)
    # out is in order K,P,phi, (conductivity, pressure, porosity)

    # pres_grad is the gradient of the pressure along the y and x directions as a tuple
    pres_grad = torch.gradient(out[:, 1:2], dim=(-2,-1))

    # get velocity by multiplying the gradient by the conductivity
    y_grad = pres_grad[0] * out[:, 0:1]
    x_grad = pres_grad[1] * out[:, 0:1]

    # compute the divergence by the second derivative of the gradients and adding them together
    yy_grad = torch.gradient(y_grad, spacing=(1,),dim=(-2,))[0]
    xx_grad = torch.gradient(x_grad, spacing=(1,),dim=(-1,))[0]
    final = yy_grad + xx_grad

    # total divergence should be 0
    loss = (final**2).mean()

    return loss, out

def train(model,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optim: Optimizer,
          crit = nn.MSELoss(),
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          loss_weights: Tuple[float,float]=(1,1)):
    
    '''
    Use only with 2-tuple output datasets

    loss_weights are MSE,Darcy\n
    for no-darcy set loss_weights[1] = 0
    '''

    epoch_loss = 0
    epoch_darcy = 0
    for feat,label in train_loader:
        optim.zero_grad()
        feat = feat.to(device)
        label = label.to(device)
        # Process darcy loss and save it
        p_loss, out = darcy_loss(model, feat)
        epoch_darcy += p_loss.item()
        # Calculate total loss
        loss = loss_weights[0] * crit(out, label) + loss_weights[1] * p_loss
        epoch_loss += loss.item()
        # Perform backward step
        loss.backward()
        optim.step()

    # Track loss
    epoch_loss /= train_loader.__len__()
    epoch_darcy /= train_loader.__len__()
    train_loss = epoch_loss
    train_darcy = epoch_darcy

    epoch_loss = 0
    epoch_darcy = 0
    with torch.no_grad():
        for feat,label in val_loader:

            feat = feat.to(device)
            label = label.to(device)
            p_loss, out = darcy_loss(model, feat)
            epoch_darcy += p_loss.item()
            loss = crit(out, label) + p_loss
            epoch_loss += loss.item()

    epoch_loss /= val_loader.__len__()
    epoch_darcy /= val_loader.__len__()

    val_loss = epoch_loss
    val_darcy = epoch_darcy

    return (train_loss, train_darcy, val_loss, val_darcy)

def train_masked(model,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optim: Optimizer,
          crit = nn.MSELoss(),
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          loss_weights: Tuple[float,float]=(1,1)):
    
    '''
    Use only with 3-tuple output datasets

    loss_weights are MSE,Darcy\n
    for no-darcy set loss_weights[1] = 0
    '''

    epoch_loss = 0
    epoch_darcy = 0
    for feat,label,mask in train_loader:
        optim.zero_grad()
        feat = feat.to(device)
        label = label.to(device)
        mask = mask.unsqueeze(1).to(device)
        # Process darcy loss and save it
        p_loss, out = darcy_loss(model, feat)
        epoch_darcy += p_loss.item()
        # Calculate total loss
        loss = crit(out * mask, label * mask) * loss_weights[0] + p_loss * loss_weights[1]
        epoch_loss += loss.item()
        # Perform backward step
        loss.backward()
        optim.step()

    # Track loss
    epoch_loss /= train_loader.__len__()
    epoch_darcy /= train_loader.__len__()
    train_loss = epoch_loss
    train_darcy = epoch_darcy

    epoch_loss = 0
    epoch_darcy = 0
    with torch.no_grad():
        for feat,label,mask in val_loader:

            feat = feat.to(device)
            label = label.to(device)
            p_loss, out = darcy_loss(model, feat)
            epoch_darcy += p_loss.item()
            loss = crit(out, label) + p_loss
            epoch_loss += loss.item()

    epoch_loss /= val_loader.__len__()
    epoch_darcy /= val_loader.__len__()

    val_loss = epoch_loss
    val_darcy = epoch_darcy

    return (train_loss, train_darcy, val_loss, val_darcy)

def weight_schedule_base(epoch: int,
                  weights: Tuple[float,float],
                  train_losses: List[float],
                  val_losses: List[float],
                  train_darcy: List[float],
                  val_darcy: List[float],
                  hyperparameters = None) -> Tuple[float,float]:
    '''
    Use this as the basis for scheduling weights
    '''
    return weights

def data_schedule_base(epoch: int,
                  args: dict,
                  train_losses: List[float],
                  val_losses: List[float],
                  train_darcy: List[float],
                  val_darcy: List[float],
                  hyperparamaters = None) -> dict:
    '''
    Use this as the basis for scheduling dataset changes
    '''
    return args.copy()

def early_stopping_base(train_losses: List[float],
                        val_losses: List[float]) -> bool:
    return False

def train_epochs(model,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optim: Optimizer,
          min_epochs: int,
          max_epochs: int,
          mask_training: bool,
          schedule = None,
          crit = nn.MSELoss(),
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          loss_weights_init: Tuple[float,float]=(1,1),
          weight_schedule: Callable = weight_schedule_base,
          early_stopping: Callable = early_stopping_base) -> Tuple[nn.Module, List[float], List[float]]:
    
    train_losses = []
    val_losses = []

    train_losses_darcy = []
    val_losses_darcy = []

    loss_weights = loss_weights_init

    for e in range(1,max_epochs+1):

        if mask_training:
            (train_loss, train_darcy, val_loss, val_darcy) = train_masked(model,train_loader,val_loader,
                                                                        optim, crit, device, loss_weights)
        else:
            (train_loss, train_darcy, val_loss, val_darcy) = train(model,train_loader,val_loader,
                                                                optim, crit, device, loss_weights)
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_losses_darcy.append(train_darcy)
        val_losses_darcy.append(val_darcy)
        
        if e > min_epochs and early_stopping(train_losses, val_losses):
            return (model, train_losses, val_losses)
        
        if schedule:
            schedule.step()

        loss_weights = weight_schedule(e, loss_weights,
                                       train_losses, val_losses,
                                       train_losses_darcy, val_losses_darcy)

    return (model, train_losses, val_losses)

def train_from_scratch(dataset_type: type[Dataset],
                       dataset_arguments: dict,
                       train_sims, val_sims,
                       optimizer: type[Optimizer],
                       min_epochs: int,
                       max_epochs: int,
                       val_steps: Tuple[int, int] = (1,201),
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       loss_weights_init: Tuple[float,float]=(1,1),
                       weight_schedule: Callable = weight_schedule_base,
                       early_stopping: Callable = early_stopping_base,
                       data_schedule: Callable = data_schedule_base,
                       data_hyperparameters = None,
                       weight_hyperparameters = None):

    args = dataset_arguments
    args["sims"] = train_sims

    train_data = dataset_type(**args)

    if len(train_data.__getitem__(0)) == 3:
        mask_training = True
    else:
        mask_training = False

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

    args["sims"] = val_sims
    args["steps"] = val_steps
    val_data = dataset_type(**args)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size= 8, shuffle=False)

    model = unet.SmallUnet().to(device)
    optim = optimizer(model.parameters())
    crit = nn.MSELoss()

    train_losses = []
    val_losses = []

    train_losses_darcy = []
    val_losses_darcy = []

    loss_weights = loss_weights_init

    for e in tqdm(range(1,max_epochs+1)):

        if mask_training:
            (train_loss, train_darcy, val_loss, val_darcy) = train_masked(model,train_loader,val_loader,
                                                                        optim, crit, device, loss_weights)
        else:
            (train_loss, train_darcy, val_loss, val_darcy) = train(model,train_loader,val_loader,
                                                                optim, crit, device, loss_weights)
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_losses_darcy.append(train_darcy)
        val_losses_darcy.append(val_darcy)
        
        if e > min_epochs and early_stopping(train_losses, val_losses):
            return (model, train_losses, val_losses)

        loss_weights = weight_schedule(e, loss_weights,
                                       train_losses, val_losses,
                                       train_losses_darcy, val_losses_darcy, weight_hyperparameters)
        
        args.pop("sims")
        new_args = data_schedule(e, args, train_losses, val_losses, train_losses_darcy, val_losses_darcy,
                                 data_hyperparameters)
        if args != new_args:
            args = new_args
            args["sims"] = train_sims
            train_data = dataset_type(**args)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size= 8, shuffle=True)
        args["sims"] = train_sims

    return (model, train_losses, val_losses)


def train_from_scratch_with_velocity(dataset_type: type[Dataset],
                       dataset_arguments: dict,
                       train_sims, val_sims,
                       optimizer: type[Optimizer],
                       min_epochs: int,
                       max_epochs: int,
                       val_steps: Tuple[int, int] = (1,201),
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       loss_weights_init: Tuple[float,float]=(1,1),
                       weight_schedule: Callable = weight_schedule_base,
                       early_stopping: Callable = early_stopping_base,
                       data_schedule: Callable = data_schedule_base,
                       data_hyperparameters = None,
                       weight_hyperparameters = None):

    args = dataset_arguments
    args["sims"] = train_sims

    train_data = dataset_type(**args)

    if len(train_data.__getitem__(0)) == 3:
        mask_training = True
    else:
        mask_training = False

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

    args["sims"] = val_sims
    args["steps"] = val_steps
    val_data = dataset_type(**args)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size= 8, shuffle=False)

    model = unet.SmallUnet().to(device)
    optim = optimizer(model.parameters())
    crit = nn.MSELoss()

    train_losses = []
    val_losses = []

    train_losses_darcy = []
    val_losses_darcy = []

    loss_weights = loss_weights_init

    for e in tqdm(range(1,max_epochs+1)):

        if mask_training:
            (train_loss, train_darcy, val_loss, val_darcy) = train_masked(model,train_loader,val_loader,
                                                                        optim, crit, device, loss_weights)
        else:
            (train_loss, train_darcy, val_loss, val_darcy) = train(model,train_loader,val_loader,
                                                                optim, crit, device, loss_weights)
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_losses_darcy.append(train_darcy)
        val_losses_darcy.append(val_darcy)
        
        if e > min_epochs and early_stopping(train_losses, val_losses):
            return (model, train_losses, val_losses)

        loss_weights = weight_schedule(e, loss_weights,
                                       train_losses, val_losses,
                                       train_losses_darcy, val_losses_darcy, weight_hyperparameters)
        
        args.pop("sims")
        new_args = data_schedule(e, args, train_losses, val_losses, train_losses_darcy, val_losses_darcy,
                                 data_hyperparameters)
        if args != new_args:
            args = new_args
            args["sims"] = train_sims
            train_data = dataset_type(**args)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size= 8, shuffle=True)
        args["sims"] = train_sims

    return (model, train_losses, val_losses)