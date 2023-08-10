#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:51:01 2020

@author: joshua
"""

""" This file contains all the custom callbacks to be used in this code. """

import sys
from torch import save as save_checkpoint
from torch.optim.lr_scheduler import _LRScheduler

class PytorchCallback(object):
    
    """ Callback base object to inherit from. """
    
    def __init__(self, name='PytorchCallback'):
        self.name = name
        
    def on_batch_end(self, *args, **kwargs):
        pass
    
    def on_epoch_end(self, *args, **kwargs):
        pass
    
    def on_train_end(self, *args, **kwargs):
        pass
    
    def on_test_end(self, *args, **kwargs):
        pass

class TerminateOnNanOrInf(PytorchCallback):
    
    """ Callback to kill the training if a Nan occurs. """
    
    def __init__(self, nan=True, inf=True, name='Terminate_On_Nan_Or_Inf'):
        super(TerminateOnNanOrInf, self).__init__(name=name)
        self.nan = nan # Check for NaNs
        self.inf = inf # Check for Infs
      
    def on_batch_end(self, *args, **kwargs):  
        if kwargs['loss'].isnan().any() and self.inf == True: # Check if the loss has gone to NaN.
            sys.exit('Loss has become NaN. Aborting training.')
        elif kwargs['loss'].isinf().any() and self.nan == True: # Check if the loss has exploded.
            sys.exit('Loss has become Inf. Aborting training.')
        
class ModelCheckpoint(PytorchCallback):
    
    """ Callback to save the model checkpoint. """
    
    def __init__(self, save_path, monitor, verbose=1, save_best_only=False, 
                 mode='auto', save_freq='epoch', save_history=False,
                 name='Model_Checkpoint'):
        super(ModelCheckpoint, self).__init__(name=name)
        
        self.save_path = save_path
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.save_history = save_history
        
        self.value = None
        
    def on_epoch_end(self, model, history, **kwargs):
        
        if self.save_freq == 'epoch':
            new_value = history[self.monitor][-1]
            if self.value is None:
                if self.verbose == 1:
                    msg = f'\n{self.monitor} improved from Inf to {new_value:.8f}. Saving checkpoint to: {self.save_path}'
                    print(msg)
                self.save(new_value, model, history, verbose=0, **kwargs)
            else:
                if self.mode == 'max':
                    if new_value > self.value:
                        self.save(new_value, model, history, **kwargs)
                    else:
                        print(f'{self.monitor} did not improve from {self.value:.8f}.')
                            
                elif self.mode == 'min':
                    if new_value < self.value:
                        self.save(new_value, model, history, **kwargs)
                    else:
                        print(f'{self.monitor} did not improve from {self.value:.8f}.')
                else:
                    if 'loss' in self.monitor.lower():
                        if new_value < self.value:
                            self.save(new_value, model, history, **kwargs)
                        else:
                            print(f'{self.monitor} did not improve from {self.value:.8f}.')
                    else:
                        if new_value > self.value:
                            self.save(new_value, model, history, **kwargs)
                        else:
                            print(f'{self.monitor} did not improve from {self.value:.8f}.')
                
        else:
            pass
        
    def on_train_end(self, model, history, **kwargs):
        last_save_path = '/'.join(self.save_path.split('/')[0:-1]) + '/last_model_checkpoint'
        msg = f'\nSaving final checkpoint to: {last_save_path}'
        print(msg)
        
        self.save(self.value, model, history, verbose=0, 
                  custom_save_path=last_save_path, **kwargs)
        
    def save(self, new_value, model, history, verbose=1, custom_save_path=None, 
             **kwargs):
        if verbose == 1:
            msg = f'\n{self.monitor} improved from {self.value:.8f} to {new_value:.8f}. Saving checkpoint to: {self.save_path}'
            print(msg)
            
        state = {'model': model.state_dict(),
                 'epoch': history['Epoch'][-1]}
        if 'optimizer' in kwargs:
            state.update({'optimizer': kwargs['optimizer'].state_dict()})
        if 'lr_schedule' in kwargs:
            if kwargs['lr_schedule'] is not None:
                state.update({'lr_schedule': kwargs['lr_schedule'].state_dict()})
        if self.save_history == True:
            state.update({'history' : history})
            
        save_checkpoint(state, self.save_path if custom_save_path is None else custom_save_path)
        self.value = new_value

class PolynomialLRDecay(_LRScheduler):
    
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, 
                 power=1.0, verbose=False):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.verbose = verbose
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_last_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
                
        if self.verbose == True:
            print(f'Learning rate changed to: {self.get_lr:.8f}')