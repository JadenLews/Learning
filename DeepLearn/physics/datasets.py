import numpy as np
import torch
import cv2

'''
This file assumes a certain file structure.
Adjust paths to your project specifications below.
'''
# Folders
BINARY_FOLDER = "../Data200x200_withinfo"
UNIFORM_FOLDER = "../Uniform200x200withInfo"

# Get porosity phi
def get_phi(sim,step,folder):
    return cv2.imread(f"{folder}/Image-{sim}-{step}_phi.jpg", cv2.IMREAD_GRAYSCALE)

# Get pressure
def get_pres(sim,step,folder):
    return cv2.imread(f"{folder}/Image-{sim}-{step}_P.jpg", cv2.IMREAD_GRAYSCALE)

# Get conductivity K
def get_k(sim,step,folder):
    return cv2.imread(f"{folder}/Image-{sim}-{step}_K.jpg", cv2.IMREAD_GRAYSCALE)

# Get all 3 as a 3-channel matrix
def get_all(sim,step,folder):
    return np.array((get_k(sim,step,folder), get_pres(sim,step,folder), get_phi(sim,step,folder)), dtype=np.float32)

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
                 channels="all",
                 future_delta=0):
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
        self.future_delta = future_delta


        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
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

        # keep step+delta inside the window
        max_step = (self.steps[1] - 1) - self.future_delta
        if sim_step > max_step:
            sim_step = max_step

        # create tensor for the target
        t_cur   = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))
        t_label = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1 + self.future_delta, folder))

        # Create 0-matrix
        z = torch.zeros_like(t_cur)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        for y0 in self.point_y:
            for x0 in self.point_x:
                disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= (self.radius**2)
                mask |= disk

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t_cur[chans, :, :], torch.zeros_like(t_cur[chans, :, :]))

        return z,t_label
    
    def __len__(self):
        return self.sims.shape[0] * self.steps * len(self.types)
    


class FixedDenseDatasetFullDelta(torch.utils.data.Dataset):
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
                 channels="all",
                 future_delta=0):
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
        self.future_delta = future_delta

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
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

        # keep step+delta inside the window
        max_step = (self.steps[1] - 1) - self.future_delta
        if sim_step > max_step:
            sim_step = max_step

        # create tensor for the target
        t_cur   = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))
        t_label = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1 + self.future_delta, folder))

        # Create 0-matrix
        z = torch.zeros_like(t_cur)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        for y0 in self.point_y:
            for x0 in self.point_x:
                disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= (self.radius**2)
                mask |= disk

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t_cur[chans, :, :], torch.zeros_like(t_cur[chans, :, :]))

        return z,t_label
    
    def __len__(self):
        return self.sims.shape[0] * self.steps * len(self.types)
    


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
                 channels="all",
                 future_delta=0):
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
        self.future_delta = future_delta

        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
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

        # keep step+delta inside the window
        max_step = (self.steps[1] - 1) - self.future_delta
        if sim_step > max_step:
            sim_step = max_step

        # create tensor for the target
        t_cur   = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))
        t_label = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1 + self.future_delta, folder))

        # Create 0-matrix
        z = torch.zeros_like(t_cur)

        # build a boolean mask of revealed pixels, shape (H,W)
        mask = torch.zeros((self.H, self.W), dtype=torch.bool)

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        for y0 in self.point_y:
            for x0 in self.point_x:
                disk = (yy - int(y0))**2 + (xx - int(x0))**2 <= (self.radius**2)
                mask |= disk

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t_cur[chans, :, :], torch.zeros_like(t_cur[chans, :, :]))

        return z,t_label
    
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
                 channels="all",
                 future_delta=0):
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
        self.future_delta = future_delta


        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
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

        # keep step+delta inside the window
        max_step = (self.steps[1] - 1) - self.future_delta
        if sim_step > max_step:
            sim_step = max_step

        # create tensor for the target
        t_cur   = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))
        t_label = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1 + self.future_delta, folder))

        # Create 0-matrix
        z = torch.zeros_like(t_cur)
        sample = torch.zeros_like(t_cur)

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
                mask |= disk
                if p in show:
                    mask_sample |= disk
                p += 1

        mask = mask.unsqueeze(0)
        mask_sample = mask_sample.unsqueeze(0)

        # write revealed pixels for selected channels
        chans = self._chan_idx()
        z[chans, :, :] = torch.where(mask, t_label[chans, :, :], torch.zeros_like(t_label[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t_cur[chans, :, :], torch.zeros_like(t_cur[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * self.steps * len(self.types)


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
                 channels="all",
                 future_delta=0):
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
        self.future_delta = future_delta


        div = (points_per_side + 1)

        self.point_x = np.arange(W // div, W, W // div)
        self.point_y = np.arange(H // div, H, H // div)

    def _chan_idx(self):
        if self.channels == "all":
            return [0,1,2]
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

        # keep step+delta inside the window
        max_step = (self.steps[1] - 1) - self.future_delta
        if sim_step > max_step:
            sim_step = max_step

        # create tensor for the target
        t_cur   = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1, folder))
        t_label = torch.tensor(get_all(self.sims[sim_idx], sim_step + 1 + self.future_delta, folder))

        # Create 0-matrix
        z = torch.zeros_like(t_cur)
        sample = torch.zeros_like(t_cur)

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
        z[chans, :, :] = torch.where(mask, t_label[chans, :, :], torch.zeros_like(t_label[chans, :, :]))
        sample[chans, :, :] = torch.where(mask_sample, t_cur[chans, :, :], torch.zeros_like(t_cur[chans, :, :]))

        return sample,z,mask
    
    def __len__(self):
        return self.sims.shape[0] * self.steps * len(self.types)