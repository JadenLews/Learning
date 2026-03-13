from typing import Tuple, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.optim import Optimizer
import datasets
import unet
from tqdm import tqdm

def physics_none(model, feat):
    out = model(feat)
    return torch.tensor(0.0, device=out.device), out

# Darcy loss function
def darcy_loss(model, inp):
    # Takes in the k,pres,phi and outputs the prediction across the image.
    inp = inp.requires_grad_(True)
    out = model(inp)
    # out is in order K,P,phi, (conductivity, pressure, porosity)

    # Impose high pressure along the entire upper line by setting the pressure channelt to 200.
    out[:, 1:2, 0, :] = 200

    # If we assume the output is in order k,pres,phi
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

def k_phi_consistency_loss(model, inp, phi_max=0.47):
    out = model(inp)
    K   = out[:, 0:1]
    phi = out[:, 2:3]

    # simulator linear mapping
    a = 2e-10 * 1e6
    b = 3e-7  * 1e6

    K_expected = a + b * (1 - phi / phi_max)

    loss = ((K - K_expected)**2).mean()
    return loss, out

def boundary_pressure_loss(model, inp):
    out = model(inp)
    P = out[:, 1:2]

    top_loss = ((P[:, :, 0, :] - 1)**2).mean()
    bottom_loss = ((P[:, :, -1, :] - 0)**2).mean()

    loss = top_loss + bottom_loss
    return loss, out

def phi_bounds_loss(model, inp, phi_max=0.47):
    out = model(inp)
    phi = out[:, 2:3]

    below_zero = torch.relu(-phi)
    above_max  = torch.relu(phi - phi_max)

    loss = (below_zero**2 + above_max**2).mean()
    return loss, out

def conductivity_positive_loss(model, inp):
    out = model(inp)
    K = out[:, 0:1]

    loss = (torch.relu(-K)**2).mean()
    return loss, out

def global_flux_loss(model, inp):
    out = model(inp)
    K = out[:, 0:1]
    P = out[:, 1:2]

    grad = torch.gradient(P, dim=(-2, -1))
    Vy = -K * grad[0]

    avg_flux = Vy.mean(dim=(-2,-1))

    loss = (avg_flux - avg_flux.mean())**2
    return loss.mean(), out

def smoothness_loss(model, inp):
    out = model(inp)
    K = out[:, 0:1]

    grad = torch.gradient(K, dim=(-2,-1))
    loss = (grad[0]**2 + grad[1]**2).mean()
    return loss, out

def physics_full(model, inp):
    darcy_l, out = darcy_loss(model, inp)
    kphi_l, _ = k_phi_consistency_loss(model, inp)
    bc_l, _ = boundary_pressure_loss(model, inp)
    phi_l, _ = phi_bounds_loss(model, inp)

    total = darcy_l + 0.5*kphi_l + 0.1*bc_l + 0.1*phi_l
    return total, out

def train(model,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optim: Optimizer,
          crit = nn.MSELoss(),
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          loss_weights: Tuple[float,float]=(1,1),
          physics_fn=None):
    
    '''
    Use only with 2-tuple output datasets

    loss_weights are MSE,Darcy\n
    for no-darcy set loss_weights[1] = 0
    '''
    if physics_fn is None:
        physics_fn = physics_none

    epoch_loss = 0.0
    epoch_phys = 0.0

    for feat,label in train_loader:
        optim.zero_grad()
        feat = feat.to(device)
        label = label.to(device)
        # Process darcy loss and save it
        p_loss, out = physics_fn(model, feat)
        # Calculate total loss
        loss = loss_weights[0] * crit(out, label) + loss_weights[1] * p_loss
        epoch_loss += loss.item()
        epoch_phys += p_loss.item()
        # Perform backward step
        loss.backward()
        optim.step()

    # Track loss
    train_loss = epoch_loss / train_loader.__len__()
    train_phys = epoch_phys / train_loader.__len__()

    epoch_loss = 0.0
    epoch_phys = 0.0
    with torch.no_grad():
        for feat,label in val_loader:
            feat = feat.to(device)
            label = label.to(device)
            p_loss, out = physics_fn(model, feat)
            loss = crit(out, label) + p_loss
            epoch_loss += loss.item()
            epoch_phys += p_loss.item()

    val_loss = epoch_loss / val_loader.__len__()
    val_phys = epoch_phys / val_loader.__len__()

    return (train_loss, train_phys, val_loss, val_phys)

def train_masked(model,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optim: Optimizer,
          crit = nn.MSELoss(),
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          loss_weights: Tuple[float,float]=(1,1),
          physics_fn=None):
    
    '''
    Use only with 3-tuple output datasets

    loss_weights are MSE,Darcy\n
    for no-darcy set loss_weights[1] = 0
    '''

    if physics_fn is None:
        physics_fn = physics_none

    epoch_loss = 0
    epoch_phys = 0
    for feat,label,mask in train_loader:
        optim.zero_grad()
        feat = feat.to(device)
        label = label.to(device)
        mask = mask.unsqueeze(1).to(device)
        # Process darcy loss and save it
        p_loss, out = physics_fn(model, feat)
        epoch_phys += float(p_loss.detach().cpu())
        # Calculate total loss
        loss = crit(out * mask, label * mask) * loss_weights[0] + p_loss * loss_weights[1]
        epoch_loss += loss.item()
        # Perform backward step
        loss.backward()
        optim.step()

    # Track loss

    train_loss = epoch_loss / train_loader.__len__()
    train_phys = epoch_phys / train_loader.__len__()

    epoch_loss = 0.0
    epoch_phys = 0.0
    with torch.no_grad():
        for feat,label,mask in val_loader:
            feat = feat.to(device)
            label = label.to(device)
            p_loss, out = physics_fn(model, feat)
            loss = crit(out, label) + p_loss
            
            epoch_loss += loss.item()
            epoch_phys += p_loss.item()

    val_loss = epoch_loss / val_loader.__len__()
    val_phys = epoch_phys / val_loader.__len__()

    return (train_loss, train_phys, val_loss, val_phys)

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
                       physics_fn=None):

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

    train_losses_phys = []
    val_losses_phys = []

    loss_weights = loss_weights_init

    for e in tqdm(range(1,max_epochs+1)):
# choose physics fn (default to "no physics")
        if physics_fn is None:
            physics_fn = physics_none

        if mask_training:
            (train_loss, train_phys, val_loss, val_phys) = train_masked(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optim=optim,
                crit=crit,
                device=device,
                loss_weights=loss_weights,
                physics_fn=physics_fn
            )
        else:
            (train_loss, train_phys, val_loss, val_phys) = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optim=optim,
                crit=crit,
                device=device,
                loss_weights=loss_weights,
                physics_fn=physics_fn
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_losses_phys.append(train_phys)
            val_losses_phys.append(val_phys)
        
        if e > min_epochs and early_stopping(train_losses, val_losses):
            return (model, train_losses, val_losses)

        loss_weights = weight_schedule(e, loss_weights,
                                       train_losses, val_losses,
                                       train_losses_phys, val_losses_phys)
        
        args.pop("sims")
        new_args = data_schedule(e, args, train_losses, val_losses, train_losses_phys, val_losses_phys,
                                 data_hyperparameters)
        if args != new_args:
            args = new_args
            args["sims"] = train_sims
            train_data = dataset_type(**args)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size= 8, shuffle=True)
        args["sims"] = train_sims

    return (model, train_losses, val_losses)