import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
class lr_scheduler_step():
    def __init__(self,Sittings,loader):
        self.initial_lr = Sittings["initial_lr"]
        self.peak_lr = Sittings["peak_lr"]
        self.warmup_ratio=Sittings["warmup"]
        if isinstance(loader,DataLoader):
            self.loader_len=len(loader)
        elif isinstance(loader,int):
            self.loader_len=loader
        else:
            raise TypeError (f"loader should be instance of DataLoader or integer")
        
        self.total_steps = self.loader_len * Sittings["num_epochs"]
        self.warmup_steps = int(self.warmup_ratio * self.total_steps)
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps
        self.track_lrs=[]
        self.lr=None
    def CheckGlobal(self,global_step):
        if global_step < self.warmup_steps:            
            self.lr = self.initial_lr + global_step * self.lr_increment
        else:
            self.lr = self.peak_lr
        self.track_lrs.append(self.lr)

                    
        
    def step(self,optimizer):
        if isinstance(optimizer,Optimizer):
            for param_group in optimizer.param_groups:   
                param_group["lr"] = self.lr

    def get_lr(self):
        return self.track_lrs[-1]

    def state_dict(self):
        return {
            "initial_lr": self.initial_lr,
            "peak_lr": self.peak_lr,
            "warmup_ratio": self.warmup_ratio,
            "loader_len": self.loader_len,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "lr_increment": self.lr_increment,
            "lr": self.lr,
            "track_lrs": self.track_lrs.copy(),
        }

    def load_state_dict(self, state_dict):
        self.initial_lr = state_dict["initial_lr"]
        self.peak_lr = state_dict["peak_lr"]
        self.warmup_ratio = state_dict["warmup_ratio"]
        self.loader_len = state_dict["loader_len"]
        self.total_steps = state_dict["total_steps"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.lr_increment = state_dict["lr_increment"]
        self.lr = state_dict["lr"]
        self.track_lrs = state_dict["track_lrs"].copy()
