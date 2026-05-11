import numpy as np
import torch
import cv2

'''
This file assumes a certain file structure.
Adjust paths to your project specifications below.
'''
# Folders
BINARY_FOLDER = "../Data200x200_withInfo_Deterministic/"
#BINARY_FOLDER = "../Data200x200_withInfo_Deterministic/Data200x200_withInfo_Deterministic/"
UNIFORM_FOLDER = "../Uniform200x200withInfo/Uniform200x200withInfo/"
BINARY_FOLDER = "../Data200x200_withInfo_Deterministic/"
#BINARY_FOLDER = "../Data200x200_withInfo_Deterministic/Data200x200_withInfo_Deterministic/"


# Get porosity phi
def get_phi(sim,step,folder):
    return (cv2.imread(f"{folder}/Sim-{sim}-Step-{step}.png").transpose(2,0,1).astype(np.float32)[2] - 128) / 128

# Get pressure
def get_pres(sim,step,folder):
    return (cv2.imread(f"{folder}/Sim-{sim}-Step-{step}.png").transpose(2,0,1).astype(np.float32)[1] - 128) / 128

# Get conductivity K
def get_k(sim,step,folder):
    return (cv2.imread(f"{folder}/Sim-{sim}-Step-{step}.png").transpose(2,0,1).astype(np.float32)[0] - 128) / 128

# Get all 3 as a 3-channel matrix
def get_all(sim,step,folder):
    return (cv2.imread(f"{folder}/Sim-{sim}-Step-{step}.png").transpose(2,0,1).astype(np.float32) - 128) / 128

def get_vel(sim,step,folder):
    return (cv2.imread(f"{folder}/Sim-{sim}-Step-{step}_Vxy.png").transpose(2,0,1).astype(np.float32) - 128) / 128

'''
Actual Datasets below
'''

class FixedDenseDatasetFull(torch.utils.data.Dataset):
    '''
    FixedDenseDatasetFull - 
    Always use this when testing
    
    Comprehensive dataset\n
    Dense means it iterates through all possible training samples in each epoch\n
    Full means the target is the full image\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = (index % self.num_steps()) + self.steps[0]

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        z = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        for y0 in self.point_y:
            for x0 in self.point_x:
                disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= (self.radius**2)
                mask |= disk

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return z,t

    def __len__(self):
        return self.sims.shape[0] * self.num_steps() * len(self.types)
    


class FixedThinDatasetFull(torch.utils.data.Dataset):
    '''
    FixedThinDatasetFull - 
    This is a weak dataset and should only be used for training. 
    
    Fast dataset\n
    Thin means it iterates through samples, but selects a random step instead of training on all steps\n
    Full means the target is the full image\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0]
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = index
        sim_step = np.random.randint(self.steps[0], self.steps[1]+1)

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        z = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        for y0 in self.point_y:
            for x0 in self.point_x:
                disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= (self.radius**2)
                mask |= disk

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return z,t
    
    def __len__(self):
        return self.sims.shape[0] * len(self.types)


class FixedDenseDatasetLimited(torch.utils.data.Dataset):
    '''
    FixedDenseDatasetLimited - 
    Randomness makes this unsuited to testing

    At each step it reveals a random assortment of the available points.
    
    Comprehensive dataset\n
    Dense means it iterates through all possible training samples in each epoch\n
    Limited means a mask is returned so that training only takes place over the given points\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        Instead of each element being (features,target) it will be (features,target,mask)

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        wiggle is the number of pixels each sensor point is shifted. Applies to each point independently

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = (index % self.num_steps()) + self.steps[0]

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        z = torch.zeros_like(t)
        sample = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)
        mask_sample = torch.zeros((self.H, self.W), dtype=torch.bool)

        show = np.random.choice(self.points_per_side**2,size=(self.points_per_side - 1)**2, replace=False)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        p = 0
        for y0 in self.point_y:
            for x0 in self.point_x:
                wigglex = np.random.randint(self.wiggle * -1, self.wiggle+1)
                wiggley = np.random.randint(self.wiggle * -1, self.wiggle+1)
                disk = (yy - int(y0+wiggley))**2 + (xx - int(x0+wigglex))**2 <= (self.radius**2)
                if p in show:
                    mask_sample |= disk
                else:
                    mask |= disk
                p += 1

        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * self.num_steps() * len(self.types)

class FixedThinDatasetLimited(torch.utils.data.Dataset):
    '''
    FixedDenseDatasetLimited - 
    Randomness makes this unsuited to testing

    At each step it reveals a random assortment of the available points.
    
    Thin dataset\n
    Thin means it iterates through samples, but selects a random step instead of training on all steps\n
    Limited means a mask is returned so that training only takes place over the given points\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        Instead of each element being (features,target) it will be (features,target,mask)

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        wiggle is the number of pixels each sensor point is shifted. Applies to each point independently

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0]:
            index = index - self.sims.shape[0]
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = index
        sim_step = np.random.randint(self.steps[0], self.steps[1])

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        z = torch.zeros_like(t)
        sample = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)
        mask_sample = torch.zeros((self.H, self.W), dtype=torch.bool)

        show = np.random.choice(self.points_per_side**2,size=(self.points_per_side - 1)**2, replace=False)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        p = 0
        for y0 in self.point_y:
            for x0 in self.point_x:
                wigglex = np.random.randint(self.wiggle * -1, self.wiggle+1)
                wiggley = np.random.randint(self.wiggle * -1, self.wiggle+1)
                disk = (yy - int(y0+wiggley))**2 + (xx - int(x0+wigglex))**2 <= (self.radius**2)
                if p in show:
                    mask_sample |= disk
                else:
                    mask |= disk
                p += 1

        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * len(self.types)

class RandomDenseDatasetLimited(torch.utils.data.Dataset):
    '''
    FixedDenseDatasetLimited - 
    Randomness makes this unsuited to testing

    At each step it reveals a random assortment of the available points.
    
    Comprehensive dataset\n
    Dense means it iterates through all possible training samples in each epoch\n
    Limited means a mask is returned so that training only takes place over the given points\n
    Random means sensor points are fully random
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 span_x = (20,180),
                 span_y = (20,180),
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        Instead of each element being (features,target) it will be (features,target,mask)

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        span_x and span_y indicate where the random points can appear.

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.span_x = span_x
        self.span_y = span_y
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = (index % self.num_steps()) + self.steps[0]

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        z = torch.zeros_like(t)
        sample = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)
        mask_sample = torch.zeros((self.H, self.W), dtype=torch.bool)

        show = np.random.choice(self.points_per_side**2,size=(self.points_per_side - 1)**2, replace=False)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        p = 0
        for y0 in self.point_y:
            for x0 in self.point_x:
                wigglex = np.random.randint(self.span_x[0], self.span_x[1])
                wiggley = np.random.randint(self.span_y[0], self.span_y[1])
                disk = (yy - int(wiggley))**2 + (xx - int(wigglex))**2 <= (self.radius**2)
                mask |= disk
                if p in show:
                    mask_sample |= disk
                p += 1

        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * self.num_steps() * len(self.types)
    

class BorderDenseDatasetLimited(torch.utils.data.Dataset):
    '''
    BorderDenseDatasetLimited - 

    At each step it reveals Boundary Conditions
    
    Comprehensive dataset\n
    Dense means it iterates through all possible training samples in each epoch\n
    Limited means a mask is returned so that training only takes place over the given points\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        Instead of each element being (features,target) it will be (features,target,mask)

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        wiggle is the number of pixels each sensor point is shifted. Applies to each point independently

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = (index % self.num_steps()) + self.steps[0]

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        z = torch.zeros_like(t)
        sample = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)
        mask_sample = torch.zeros((self.H, self.W), dtype=torch.bool)

        mask[0:5,:] = 1
        mask[-5:,:] = 1
        mask[10:15,:] = 1

        mask[:,99:101] = 1
        mask[:,185:190] = 1
        mask[:,10:15] = 1

        mask_sample[0:5,:] = 1
        mask_sample[-5:,:] = 1

        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * self.num_steps() * len(self.types)
    

class BorderThinDatasetLimited(torch.utils.data.Dataset):
    '''
    BorderThinDatasetLimited - 

    At each step it reveals Boundary Conditions
    
    Comprehensive dataset\n
    Dense means it iterates through all possible training samples in each epoch\n
    Limited means a mask is returned so that training only takes place over the given points\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        Instead of each element being (features,target) it will be (features,target,mask)

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        wiggle is the number of pixels each sensor point is shifted. Applies to each point independently

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = np.random.randint(self.steps[0], self.steps[1])

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))[self._chan_idx()]

        # Create 0-matrix
        z = torch.zeros_like(t)
        sample = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)
        mask_sample = torch.zeros((self.H, self.W), dtype=torch.bool)

        mask[0:5,:] = 1
        mask[-5:,:] = 1
        mask[10:15,:] = 1

        mask[:,99:101] = 1
        mask[:,185:190] = 1
        mask[:,10:15] = 1

        mask_sample[0:5,:] = 1
        mask_sample[-5:,:] = 1

        #mask_sample[:,185:190] = 1
        #mask_sample[:,10:15] = 1


        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * len(self.types)
    
class BorderDenseDatasetFull(torch.utils.data.Dataset):
    '''
    BorderDenseDatasetLimited - 

    At each step it reveals Boundary Conditions
    
    Comprehensive dataset\n
    Dense means it iterates through all possible training samples in each epoch\n
    Limited means a mask is returned so that training only takes place over the given points\n
    Fixed means sensor points do not change
    '''

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all"):
        '''
        sims should be train_sims, val_sims, or test_sims

        Instead of each element being (features,target) it will be (features,target,mask)

        points_per_side are equally spaced, an there will be n^2 many points

        radius is the size of sensor points

        steps is the number of steps (low,high) to use

        types should be a list either [0],[1],[0,1]. 0 indicates binary, 1 indicates uniform.

        wiggle is the number of pixels each sensor point is shifted. Applies to each point independently

        H=200, the height of simulation space.
        
        W=200, the width of simulation space.

        channels must be 'all', 'K', 'P', or 'phi' and refers to which parts of data will be used.
        '''
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', or 'phi'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = (index % self.num_steps()) + self.steps[0]

        # Create tensor for the target
        t = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # Create 0-matrix
        sample = torch.zeros_like(t)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        mask[0:5,:] = 1
        mask[-5:,:] = 1

        #mask[:,185:190] = 1
        #mask[:,10:15] = 1


        mask = mask.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        sample[chans, :, :] = torch.where(mask, t[chans, :, :], torch.zeros_like(t[chans, :, :]))

        return sample,t
    
    def __len__(self):
        return self.sims.shape[0] * self.num_steps() * len(self.types)
    












class BorderThinPressureGradientDatasetLimited(torch.utils.data.Dataset):

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all",
                 num_pressure_points=9):
        
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types
        self.num_pressure_points = num_pressure_points

        div = (points_per_side + 1)
        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', 'phi', or 'KP'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def add_pressure_gradient_points(self, mask, t):
        # t should be full 3-channel tensor: K, P, phi
        P = t[1].numpy()

        dP_dy, dP_dx = np.gradient(P)
        grad_mag = np.sqrt(dP_dx**2 + dP_dy**2)

        # avoid choosing border pixels, since border is already revealed
        grad_mag[:10, :] = 0
        grad_mag[-10:, :] = 0
        grad_mag[:, :10] = 0
        grad_mag[:, -10:] = 0

        flat_idx = np.argpartition(
            grad_mag.flatten(),
            -self.num_pressure_points
        )[-self.num_pressure_points:]

        ys, xs = np.unravel_index(flat_idx, grad_mag.shape)

        yy, xx = torch.meshgrid(
            torch.arange(self.H),
            torch.arange(self.W),
            indexing="ij"
        )

        for y0, x0 in zip(ys, xs):
            disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= self.radius**2
            mask |= disk

        return mask

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = np.random.randint(self.steps[0], self.steps[1])

        # keep full t first because pressure is always channel 1
        t_full = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        # then select channels for actual output
        chans = self._chan_idx()
        t = t_full[chans]

        z = torch.zeros_like(t)
        sample = torch.zeros_like(t)

        mask = torch.zeros((self.H, self.W), dtype=torch.bool)
        mask_sample = torch.zeros((self.H, self.W), dtype=torch.bool)

        # same border code as before
        mask[0:5,:] = 1
        mask[-5:,:] = 1
        mask[10:15,:] = 1

        mask[:,99:101] = 1
        mask[:,185:190] = 1
        mask[:,10:15] = 1

        mask_sample[0:5,:] = 1
        mask_sample[-5:,:] = 1

        # add dynamic pressure-gradient interior points
        mask = self.add_pressure_gradient_points(mask, t_full)

        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # since t already has selected channels, use all selected output channels
        z[:, :, :] = torch.where(mask, t, torch.zeros_like(t))
        sample[:, :, :] = torch.where(mask_sample, t, torch.zeros_like(t))

        return sample, z, mask
    
    def __len__(self):
        return self.sims.shape[0] * len(self.types)
    





class BorderDensePressureGradientDatasetFull(torch.utils.data.Dataset):

    def __init__(self,
                 sims,
                 points_per_side = 3,
                 radius = 5,
                 steps = (0,200),
                 types=[0],
                 wiggle=0,
                 H=200,
                 W=200,
                 channels="all",
                 num_pressure_points=9):
        
        self.sims = sims
        self.n_sims = sims.shape[0]
        self.points_per_side = points_per_side
        self.steps = steps
        self.radius = radius
        self.H, self.W = H, W
        self.wiggle = wiggle
        self.channels = channels
        self.types = types
        self.num_pressure_points = num_pressure_points

        div = (points_per_side + 1)
        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
        if self.channels == "KP":
            return [0,1]
        elif self.channels == "K":
            return [0]
        elif self.channels == "P":
            return [1]
        elif self.channels == "phi":
            return [2]
        else:
            raise ValueError("channels must be 'all', 'K', 'P', 'phi', or 'KP'")
        
    def num_steps(self):
        return self.steps[1] - self.steps[0]

    def add_pressure_gradient_points(self, mask, t):
        P = t[1].numpy()

        dP_dy, dP_dx = np.gradient(P)
        grad_mag = np.sqrt(dP_dx**2 + dP_dy**2)

        grad_mag[:10, :] = 0
        grad_mag[-10:, :] = 0
        grad_mag[:, :10] = 0
        grad_mag[:, -10:] = 0

        flat_idx = np.argpartition(
            grad_mag.flatten(),
            -self.num_pressure_points
        )[-self.num_pressure_points:]

        ys, xs = np.unravel_index(flat_idx, grad_mag.shape)

        yy, xx = torch.meshgrid(
            torch.arange(self.H),
            torch.arange(self.W),
            indexing="ij"
        )

        for y0, x0 in zip(ys, xs):
            disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= self.radius**2
            mask |= disk

        return mask

    def __getitem__(self, index):

        if index >= self.sims.shape[0] * self.num_steps():
            index = index - self.sims.shape[0] * self.num_steps()
            kind = self.types[1]
        else:
            kind = self.types[0]

        if kind == 0:
            folder = BINARY_FOLDER
        else:
            folder = UNIFORM_FOLDER

        sim_idx = (index // self.num_steps()) % self.n_sims
        sim_step = (index % self.num_steps()) + self.steps[0]

        t_full = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))

        chans = self._chan_idx()
        t = t_full[chans]

        sample = torch.zeros_like(t)

        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        # same border code as before
        mask[0:5,:] = 1
        mask[-5:,:] = 1

        # add dynamic pressure-gradient interior points
        mask = self.add_pressure_gradient_points(mask, t_full)

        mask = mask.unsqueeze(0)

        sample[:, :, :] = torch.where(mask, t, torch.zeros_like(t))

        return sample, t
    
    def __len__(self):
        return self.sims.shape[0] * self.num_steps() * len(self.types)