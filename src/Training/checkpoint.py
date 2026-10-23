
# src/training/checkpoint.py

from pathlib import Path

import torch
import random
import numpy as np

    
def get_rng_state():
    return {
        'python': random.getstate(),
         'numpy':np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None),
            }


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])

'''
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    batch_idx,
    global_step,
    rn_gnt,
    sampler,
    scheduler=None,
    train_loss=None,
):
    #Save model weights and training progress.

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state':rn_gnt,
        'train_loss': train_loss,
        'batch_idx': batch_idx
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if sampler is not None:
            checkpoint['sampler_state_dict'] = sampler.state_dict()

    # Write to a temporary file before replacing the target.
    temp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def load_checkpoint(
    path,
    model,
    optimizer,
    device=None,
    scheduler=None,
    sampler=None,
    rng=None
):

    #Restore a checkpoint and return the training progress.

    checkpoint = torch.load(
        path,
        map_location='cpu',
        weights_only=True,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        if 'scheduler_state_dict' not in checkpoint:
            raise ValueError('Checkpoint does not contain scheduler state.')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    
    if rng is not None:
        if 'rng_state' not in checkpoint:
            raise ValueError('Checkpoint does not contain random number generator state.')
        rng(checkpoint['rng_state'])
    if sampler is not None:
            if 'sampler_state_dict' not in checkpoint:
                raise ValueError('Checkpoint does not contain sampler state.')
            sampler.load_state_dict(checkpoint['sampler_state_dict'])
    

    return {
        'start_epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'batch_idx':checkpoint['batch_idx'],
                #'train_loss': checkpoint.get('train_loss'),

    }

'''






class Save_checkpoint:


    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.model=None
        self.optimizer=None
        self.epoch=None
        self.batch_idx=None
        self.global_step=None
        self.rn_gnt=None
        self.sampler=None
        self.scheduler=None
        self.train_loss=None
        self.epoch_completed=False
        self.lora_state=None
        self.training_time=None

    def seve_all(self,model,optimizer,epoch,batch_idx,
                 global_step,sampler,lora=False,scheduler=None,
                 train_loss=None,epoch_completed=False):
        checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'epoch_completed': epoch_completed,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': get_rng_state(),
                'train_loss': train_loss,
                'epoch_completed':epoch_completed
            }
        if not lora:
            checkpoint['model_state_dict']= model.state_dict()

        if lora :
            self.set_lora_only(model)
            checkpoint['lora_state']=self.lora_state
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
        if sampler is not None:
            checkpoint['sampler_state_dict'] = sampler.state_dict()
    
        temp_path = self.path.with_suffix(self.path.suffix + '.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(self.path)
    
    def set_model(self,model):
        self.model=model

    def set_lora_only(self,model):
        self.lora_state = {name: param.detach().cpu() for name, param in model.state_dict().items() if '.lora.A' in name or '.lora.B' in name}

    def set_optimizer(self,optimizer):
        self.optimizer=optimizer

    def set_sampler(self,sampler):
        self.sampler=sampler
    def set_scheduler(self,scheduler):
        self.scheduler=scheduler

    def set_epoch_completed(self,epoch_completed):
        self.epoch_completed=epoch_completed

    def set_epoch(self,epoch):
        self.epoch=epoch
    
    def set_batch_idx(self,batch):
        self.batch_idx=batch
    def set_global_step(self,global_step):
        self.global_step=global_step

    def set_rng(self):
        self.rn_gnt=get_rng_state()

    def set_train_loss(self,train_loss):
        self.train_loss=train_loss

    def set_train_time(self,time):
        self.training_time=time

    def save_checkpoint(self):
        checkpoint = {}
    
        
        if self.epoch is not None:
            checkpoint['epoch'] = self.epoch        
        if self.epoch_completed is not None:
            checkpoint['epoch_completed'] = self.epoch_completed
        if self.batch_idx is not None:
                    checkpoint['batch_idx'] = self.batch_idx       
        if self.global_step is not None:
                checkpoint['global_step'] = self.global_step
        if self.model is not None:
            checkpoint['model_state_dict'] = self.model.state_dict() 
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.train_loss is not None:
            checkpoint['train_loss'] = self.train_loss

        if self.rn_gnt is not None:
            checkpoint['rng_state'] = self.rn_gnt
                
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
    
        if self.sampler is not None:
            checkpoint['sampler_state_dict'] = self.sampler.state_dict()
        if self.lora_state is not None:
            checkpoint['lora_state'] = self.lora_state

        if self.training_time is not None:
            checkpoint['total_training_time'] = self.training_time

    
        temp_path = self.path.with_suffix(self.path.suffix + '.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(self.path)






class Load_Checkpoint:
    def __init__(self,path,device='cpu'):
         self.checkpoint=torch.load(
                     path,
                     map_location=device,
                     weights_only=False,
                 )
    
    def load_model(self,model):
        model.load_state_dict(self.checkpoint['model_state_dict'])

    def load_optimizer(self,optimizer):
        optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def load_scheduler(self,scheduler):
        if scheduler is not None:
            if 'scheduler_state_dict' not in self.checkpoint:
                raise ValueError('Checkpoint does not contain scheduler state.')
            scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

    def load_sampler(self,sampler):
        if sampler is not None:
            if 'sampler_state_dict' not in self.checkpoint:
                raise ValueError('Checkpoint does not contain sampler state.')
            sampler.load_state_dict(self.checkpoint['sampler_state_dict'])

    def load_rng(self):
        if 'rng_state' not in self.checkpoint:
            raise ValueError('Checkpoint does not contain random number generator state.')
        set_rng_state(self.checkpoint['rng_state'])
    def load_lora_only(self,model):
            if 'lora_state' not in self.checkpoint:
                raise ValueError('Checkpoint does not contain LoRA state.')
            model.load_state_dict(self.checkpoint['lora_state'], strict=False)

    def return_rest(self):
        epoch = self.checkpoint['epoch']
        completed = self.checkpoint.get('epoch_completed', False)

        if completed:
            start_epoch = epoch + 1
            batch_idx = 0
        else:
            start_epoch = epoch
            batch_idx = self.checkpoint.get('batch_idx', -1)+1

        return {
            'start_epoch': start_epoch,
            'batch_idx': batch_idx,
            'global_step': self.checkpoint['global_step'],
            'total_training_time':self.checkpoint['total_training_time']

        }
        """    
        def return_rest(self):
                return {
                            'start_epoch': self.checkpoint['epoch'],
                            'global_step': self.checkpoint['global_step'],
                            'batch_idx':self.checkpoint['batch_idx'],
                                    #'train_loss': self.checkpoint.get('train_loss'),
                        }"""




