#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:56:30 2020

@author: joshua
"""

import numpy as np
import time
import torch

from callbacks import TerminateOnNanOrInf

def incremental_average(x1, x2, n):
    return x1 + (x2-x1)/n

class PytorchModel:
    
    """ A custom training class to manage the training of D-Net. Structured
    similar to the 'Model' class in Keras as it features functions like 'fit',
    and 'test'. """
    
    def __init__(self, model, optimizer=None, loss=None, loss_weights=None, 
                 metrics=None, lr_schedule=None, history=None, model_name=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.loss = loss
        self.loss_weights = loss_weights
        self.metrics = metrics
        self.model_name = model_name
        
        self.total_loss = 0   
        if history is None:
            self.history = {'Epoch' : [], 'Epoch_Duration' : []}
        else:
            self.history = history

    def fit(self, epochs, batch_size, train_dataset, train_steps, 
            initial_epoch=0, valid_dataset=None, valid_steps=None,
            callbacks_list=[TerminateOnNanOrInf], loss_acc_accumulate=500,
            apply_fn=None, save_on_completion=False):
        
        def train_step(x, y, progress_data={}):
            
            # This line of code places the data onto the gpu. This must be call inside the training step otherwise
            # the memory allocation remains meaning gradually GPU resources are eaten up.
            x = {key : torch.autograd.Variable(val.cuda()) for key, val in x.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
            y = {key : torch.autograd.Variable(val.cuda()) for key, val in y.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
            
            self.optimizer.zero_grad()
            
            total_loss = 0
            losses = {}
            
            with torch.cuda.amp.autocast():
                logits = self.model(x)  # Logits for this minibatch (May have multiple outputs).
  
                # Compute the loss value for this minibatch.
                if type(self.loss) == dict:
                    for loss_key, loss_obj in self.loss.items():
                        loss = loss_obj.forward(y[loss_key], logits[loss_key]) # Update the loss for a given output.
                        loss *= 1 if self.loss_weights is None else self.loss_weights[loss_key] # Apply the loss weights where appropriate.
                        losses[loss_key+'_Loss'] = loss.item() # Update the loss values where expected.
                        total_loss += loss # Add this to the total loss.
                else:
                    total_loss = self.loss.forward(y['depth'], logits['depth'])
                losses['Loss'] = total_loss.item()
                
            grad_scalar.scale(total_loss).backward()
            grad_scalar.step(self.optimizer)
            if self.lr_schedule is not None:
                self.lr_schedule.step()
            
            for metric_key, metric_values in self.metrics.items(): # Make the metrics as a class that has update_state as a function.
                if type(metric_values) == dict:
                    for metric in metric_values:
                        metric.compute(y[metric_key].detach(), logits[metric_key].detach())
                else:
                    metric_values.compute(y['depth'].detach(), logits['depth'].detach())
                    
            for loss_key, loss_value in losses.items(): # Collect the loss values for display.
                progress_data[loss_key] = incremental_average(progress_data[loss_key] if step > 0 else 0, 
                                                              loss_value, step+1 if step+1 < loss_acc_accumulate else loss_acc_accumulate)
                
            grad_scalar.update()
            
            # Apply all on batch end callbacks here.
            for callback in callbacks_list:
                callback.on_batch_end(self.model, history=self.history, 
                                      optimizer=self.optimizer, 
                                      lr_schedule=self.lr_schedule,
                                      loss=total_loss, training=True)
                
            return progress_data
        
        def valid_step(x, y, progress_data={}):

            # This line of code places the data onto the gpu. 
            x = {key : torch.autograd.Variable(val.cuda()) for key, val in x.items()} # This sends the data to the GPU.
            y = {key : torch.autograd.Variable(val.cuda()) for key, val in y.items()} # This sends the data to the GPU.
        
            total_loss = 0
            losses = {}
            
            with torch.cuda.amp.autocast():
                logits = self.model(x)  # Logits for this minibatch.
  
                # Compute the loss value for this minibatch.
                if type(self.loss) == dict:
                    for loss_key, loss_obj in self.loss.items():
                        loss = loss_obj.forward(y[loss_key], logits[loss_key]) # Update the loss for a given output.
                        loss *= 1 if self.loss_weights is None else self.loss_weights[loss_key] # Apply the loss weights where appropriate.
                        losses[loss_key+'_Val_Loss'] = loss.item() # Update the loss values where expected.
                        total_loss += loss # Add this to the total loss.
                else:
                    total_loss = self.loss.forward(y['depth'], logits['depth'])
                losses['Val_Loss'] = total_loss.item()
            
            for metric_key, metric_values in self.metrics.items(): # Make the metrics as a class that has update_state as a function.
                if type(metric_values) == dict:
                    for metric in metric_values:
                        metric.compute(y[metric_key].detach(), logits[metric_key].detach())
                else:
                    metric_values.compute(y['depth'].detach(), logits['depth'].detach())
                    
            for loss_key, loss_value in losses.items(): # Collect the loss values for display.
                progress_data[loss_key] = incremental_average(progress_data[loss_key] if step > 0 else 0, 
                                                              loss_value, step+1)
                
            # Apply all on batch end callbacks here.
            for callback in callbacks_list:
                callback.on_batch_end(self.model, history=self.history, 
                                      optimizer=self.optimizer, 
                                      lr_schedule=self.lr_schedule,
                                      loss=total_loss, training=False)
                    
            return progress_data
        
        # Throw everything into training mode.
        self.model.train()
        for metric in self.metrics.values():
            metric.train() 
            
        # Set the learning rate_schedule to the correct step.
        if self.lr_schedule is not None:
            self.history['Learning_Rate'] = [self.lr_schedule.get_last_lr()[-1]]
            print(f"\nLearning rate is initialized to: {self.history['Learning_Rate'][-1]:.8f}\n")
            
        if apply_fn is not None:
            apply_fn(self.model, verbose=1)
            
        train_bar = Progress_Bar(int(np.ceil(train_steps/batch_size)), "Training",
                                 end_msg="Training epoch complete!")
        
        if valid_dataset is not None:
            valid_bar = Progress_Bar(valid_steps, "Validating",
                                     end_msg="Validation epoch complete!")
        
        grad_scalar = torch.cuda.amp.GradScaler()
        
        for metric in self.metrics.values():
            metric.accumulate_limit = loss_acc_accumulate
        
        if valid_dataset is None:
            print("\nTraining for {0} steps with batch size {1}.".format(int(np.ceil(train_steps/batch_size)), batch_size))
        else:
            print("\nTraining for {0} steps with batch size {1}.\nValidating for {2} steps with batch size 1.".format(int(np.ceil(train_steps/batch_size)), batch_size, valid_steps))
          
        
        for epoch in range(initial_epoch, epochs):
            print("\nEpoch %d/%d" % (epoch+1, epochs))

            train_progress_data = {}
            train_bar.reset_progress()
            train_bar.start_progress()
    
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset.data):
                        
                if step < int(np.ceil(train_steps/batch_size)):
                    train_progress_data = train_step(x_batch_train, 
                                                     y_batch_train, 
                                                     train_progress_data)
                else:
                    break
                
                for metric in self.metrics.values():
                    train_progress_data[metric.name] = metric.result()
    
                train_bar.update_progress(step+1, 
                                          new_display_data=train_progress_data)

            train_bar.finish_progress(verbose=False)
                
            # Reset training metrics at the end of each epoch
            for metric in self.metrics.values():
                metric.reset()
            
            if valid_dataset is not None:
                valid_progress_data = {}
                valid_bar.reset_progress()    
                valid_bar.start_progress()
                
                self.model.eval() # Throw model into eval and test mode.
                for metric in self.metrics.values():
                    metric.test()
            
                with torch.no_grad(): # Disable gradient calculations for test performance.
                    # Run a validation loop at the end of each epoch.
                    for step, (x_batch_val, y_batch_val) in enumerate(valid_dataset.data):
                        
                        if step < valid_steps:
                            valid_progress_data = valid_step(x_batch_val, 
                                                             y_batch_val, 
                                                             valid_progress_data)
    
                        else:
                            break
                        
                        # Update the metric results and reset states.
                        for metric in self.metrics.values():
                            valid_progress_data['Val_'+metric.name] = metric.result()
                            
                        valid_bar.update_progress(step+1, 
                                                  new_display_data=valid_progress_data)
            
                valid_bar.finish_progress() # We don't want to print anymore. 
                
                for metric in self.metrics.values():
                    metric.reset()
                    
                # Throw everything back into training mode.
                self.model.train()
                for metric in self.metrics.values():
                    metric.train() 
                if apply_fn is not None:
                    apply_fn(self.model)
            
            # Update the history.
            self.history['Epoch'].append(epoch)
            for key, value in train_progress_data.items():
                if key in self.history:
                    self.history[key].append(value)
                else:
                    self.history[key] = [value]
            for key, value in valid_progress_data.items():
                if key in self.history:
                    self.history[key].append(value)
                else:
                    self.history[key] = [value]
            if self.lr_schedule is not None:
                self.history['Learning_Rate'].append(self.lr_schedule.get_last_lr()[-1])
                print(f"Learning rate has been adjusted to: {self.history['Learning_Rate'][-1]:.8f}\n")
                    
            # Run the epoch end callbacks.
            for callback in callbacks_list:
                callback.on_epoch_end(self.model, optimizer=self.optimizer, 
                                      lr_schedule=self.lr_schedule,
                                      history=self.history)
                
        for callback in callbacks_list:
            callback.on_train_end(self.model, optimizer=self.optimizer, 
                                  lr_schedule=self.lr_schedule,
                                  history=self.history)
        print('\nTraining complete!')
                
    def test(self, test_dataset, test_steps, 
             callbacks_list=[]):
        
        def test_step(x, y, progress_data={}):

            # This line of code places the data onto the gpu. 
            x = {key : torch.autograd.Variable(val.cuda()) for key, val in x.items()} # This sends the data to the GPU. 
            y = {key : torch.autograd.Variable(val.cuda()) for key, val in y.items()} # This sends the data to the GPU. 
        
            total_loss = 0
            losses = {}
            
            with torch.cuda.amp.autocast():
                logits = self.model(x)  # Logits for this minibatch (May have multiple outputs).
  
                # Compute the loss value for this minibatch.
                if type(self.loss) == dict:
                    for loss_key, loss_obj in self.loss.items():
                        loss = loss_obj.forward(y[loss_key], logits[loss_key]) # Update the loss for a given output.
                        loss *= 1 if self.loss_weights is None else self.loss_weights[loss_key] # Apply the loss weights where appropriate.
                        losses[loss_key+'_Loss'] = loss.item() # Update the loss values where expected.
                        total_loss += loss # Add this to the total loss.
                else:
                    total_loss = self.loss.forward(y['depth'], logits['depth'])
                losses['Loss'] = total_loss.item()
            
            for metric_key, metric_values in self.metrics.items(): # Make the metrics as a class that has update_state as a function.
                if type(metric_values) == dict:
                    for metric in metric_values:
                        metric.compute(y[metric_key].detach(), logits[metric_key].detach())
                else:
                    # metric_values.evaluate(y_batch_train['depth'].detach().cpu().squeeze(), logits['depth'].detach().cpu().squeeze())
                    metric_values.compute(y['depth'].detach(), logits['depth'].detach())
                    
            for loss_key, loss_value in losses.items(): # Collect the loss values for display.
                progress_data[loss_key] = incremental_average(progress_data[loss_key] if step > 0 else 0, 
                                                              loss_value, step+1)
                
            # Apply all on batch end callbacks here.
            for callback in callbacks_list:
                callback.on_batch_end(self.model, history=self.history)
                    
            return progress_data
        
        test_bar = Progress_Bar(test_steps, "Testing", end_msg="Testing complete!")
        
        self.model.eval() # Throw model into eval and test mode.
        for metric in self.metrics.values():
            metric.test()
            
        progress_data = {}
        
        test_bar.reset_progress()
        test_bar.start_progress()
    
        with torch.no_grad(): # Disable gradient calculations for test performance.
            for step, (x_batch_test, y_batch_test) in enumerate(test_dataset.data):
                
                if step < test_steps:
                    progress_data = test_step(x_batch_test, y_batch_test, progress_data)

                else:
                    break
                
                # Update the metric result.
                for metric in self.metrics.values():
                    progress_data['val_'+metric.name] = metric.result()
                    
                test_bar.update_progress(step+1, new_display_data=progress_data)
    
        test_bar.finish_progress() # We don't want to print anymore. 
        
        for metric in self.metrics.values():
            metric.reset()
            
        self.model.train() # Throw model into training mode.
        for metric in self.metrics.values():
            metric.train()
            
        # Apply all on epoch end callbacks here.
            for callback in callbacks_list:
                callback.on_epoch_end(self.model, history=self.history)
                
class Metric(object): # 'object' keyword defines this as a base class and can only be used as a base class.

    """ Metric base class to inherit from. """

    def __init__(self, accumulate_limit=100, name='Metric'):
        
        self.name = name
        self.accumulate_limit = accumulate_limit
        self.value = None
        self.eval = False
        self.n = 0
        
    def compute(self, y_true, y_pred):
        
        res = self.call(y_true, y_pred)
        if self.value is None:
            self.value = res
            if self.eval == True:
                self.n += 1
        else:
            if self.eval == False:
                self.value = self.value + (res-self.value)/self.accumulate_limit
            else:
                self.value = self.value + (res-self.value)/(self.n)
                self.n += 1
            
    def call(self, y_true, y_pred):
        pass
        
    def result(self):
        return self.value.item()
    
    def reset(self):
        self.value = None
        
    def train(self):
        self.eval = False
    
    def test(self):
        self.eval = True
        self.n = 0

def weights_init_xavier(m): # Initialise model weights using xavier initialisation.
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
def weights_init_kaiming_normal(m): # Initialise model weights using kaiming normal initialisation.
    if (isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or 
        isinstance(m, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
            
def print_summary(model, show_model=False): # Prints a summary of the model.
    
    if show_model == True:
        print(model)
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of trainable parameters: {}".format(num_params_update))
    
    print("Total number of non-trainable parameters: {}".format(num_params-num_params_update))

def special_mask(method='eigen', dataset=None): # Generates masks used from model evaluation according to standard practices for depth estimation.
    
    def mask_fn(mask, shape=None):
        
        if shape is not None and dataset is not None:
            
            eval_mask = torch.zeros_like(mask)

            if method == 'garg':
                if len(eval_mask.shape) > 2:
                    eval_mask[:, int(0.40810811*shape[1]) : int(0.99189189*shape[1]), 
                              int(0.03594771*shape[2]) : int(0.96405229*shape[2])] = 1
                else:
                    eval_mask[int(0.40810811*shape[0]) : int(0.99189189*shape[0]), 
                              int(0.03594771*shape[1]) : int(0.96405229*shape[1])] = 1
        
            elif method == 'eigen':
                if dataset == 'kitti':
                    if len(eval_mask.shape) > 2:
                        eval_mask[:, int(0.3324324*shape[1]) : int(0.91351351*shape[1]), 
                                  int(0.0359477*shape[2]) : int(0.96405229*shape[2])] = 1
                    else:
                        eval_mask[int(0.3324324*shape[0]) : int(0.91351351*shape[0]), 
                                  int(0.0359477*shape[1]) : int(0.96405229*shape[1])] = 1
                elif dataset == 'nyu':
                    if len(eval_mask.shape) > 2:
                        eval_mask[:, 45:471, 41:601] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1

            else:
                eval_mask = torch.ones_like(eval_mask)
                    
            if mask is not None:
                mask = torch.logical_and(mask, eval_mask)
            else:
                mask = eval_mask
        
        return mask
    
    return mask_fn # Take advantage of python's inner function scope references.

class Progress_Bar:
    
    """ A progress bar used to show training progress, show live stats and
    predict time remaining similar to that of Keras 'fit'. """
    
    def __init__(self, length=-1, default_status=None, format_dhms=True, 
                 end_msg='Progress Complete!'):
        
        self.length = length
        self.default_status = default_status
        self.total_time = 0
        self.t1 = 0
        self.t2 = 0
        self.average_time = 0
        self.num_terms = 0
        self.default_display_data = {'ETA' : 'Pending...',
                                     'Iteration Time' : '0.0'}
        self.longest_len = 0
        self.format_dhms = format_dhms
        self.end_msg = end_msg
        self.start_count = False
        
    def start_progress(self):
        self.t1 = time.time()
        self.default_display_data['Iteration Time'] = '0.0s'
        
        self.show_progress(step=0)
        
    def update_progress(self, step, new_display_data={}, new_status=None):
        
        def convert_time_format(x):

            secs = x
            if self.format_dhms == True: # Convert to days hours minutes seconds format for easier reading.
                if x >= 60: # Longer than a minute.
                    if x >= 3600: # Longer than an hour.
                        if x >= 86400: # Longer than a day.
                            days = int(x//86400)
                            hrs = x%86400 # Use modulus to get the hours.
                            mins = hrs%3600 # Use modulus to get the minutes.
                            hrs = int(hrs//3600) # Get the hours.
                            secs = int(mins%60) # Get the seconds without the minutes.
                            mins = int(mins)//60 # Get the minutes without the seconds.
                            msg = '{0}d {1}h {2}min {3}s'.format(days, hrs, mins, secs)
                        else:
                            hrs = int(x//3600) # Get the hours.
                            mins = x%3600 # Use modulus to get the minutes.
                            secs = int(mins%60) # Get the seconds without the minutes.
                            mins = int(mins)//60 # Get the minutes without the seconds.
                            msg = '{0}h {1}min {2}s'.format(hrs, mins, secs)
                    else:
                        secs = int(x%60) # Get the seconds without the minutes.
                        mins = int(x//60) # Get the minutes without the seconds.
                        msg = '{0}min {1}s'.format(mins, secs)
                else:
                    msg = '{:.2f}s'.format(secs)
            else:
                msg = '{:.2f}s'.format(secs)
                
            return msg
        
        self.t2 = time.time()
        time_diff = self.t2-self.t1
        
        if self.start_count == True:
            self.num_terms += 1
            self.average_time += (time_diff - self.average_time)/self.num_terms # This method ensures a running average is kept such that the time doesn't fluctuate too much.
            self.total_time += time_diff # This computes the total time. 
        else:
            self.start_count = True
        eta_time = (self.length - step) * self.average_time

        self.default_display_data['ETA'] = convert_time_format(eta_time) # Estimate the time remaing based on the time taken for each iteration averaged out.
        self.default_display_data['Iteration Time'] = convert_time_format(time_diff) # Find the time taken to process this iteration.
        
        self.show_progress(step=step, new_display_data=new_display_data,
                           new_status=new_status)
              
        self.t1 = time.time()
        
    def finish_progress(self, new_display_data={}, new_status=None, 
                        verbose=True):
        
        self.t2 = time.time()
        self.total_time += self.t2-self.t1 # This computes the total time.
        self.average_time = round(self.total_time/self.length, 2)
        
        if verbose == True:
            text = "\n\n"+self.end_msg+" \nTotal Time Elapsed: {0} - Average time per iteration: {1} - Status: {2}".format(round(self.total_time, 3), 
                                                                                                                           self.average_time, 
                                                                                                                           'Done.' if new_status is None else new_status)
            print(text+'\n')
        else:
            print('\n')
                
    def report_times(self): # Only call if process is finished.
        return self.total_time, self.average_time
        
    def reset_progress(self, new_length=None, new_default_status=None):
        if new_length is not None:
            self.length = new_length
        if new_default_status is not None:
            self.default_status = new_default_status
            
        self.total_time = 0
        self.average_time = 0
        self.num_terms = 0
        self.t1 = 0
        self.t2 = 0
        
        self.longest_len = 0
        self.start_count = False
        
    def show_progress(self, step, new_display_data={}, new_status=None):
        '''
        update_progress() : Displays or updates a console progress bar
        Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        '''
        
        progress = step/self.length
        barLength = 30 # Modify this to change the length of the progress bar
        if new_status is None:
            status = self.default_status
        else:
            status = new_status
            
        if isinstance(progress, int):
            progress = float(progress)
            
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float"
            
        if progress < 0:
            progress = 0
            status = "Halt..."
            
        if progress >= 1:
            progress = 1
            status = "Done..."
            
        block = int(round(barLength*progress))
        if (block >= 1 and block < barLength) or (block >= 1 and progress < 1):
            arrow = ">"
            block -= 1
        else:
            arrow = ""
            block += 1
          
        data_msg = ""
        display_data = {**self.default_display_data, **new_display_data}
        for key, value in display_data.items():
            if isinstance(value, str):
                data_msg += key + ": " + value + " - "
            else:
                data_msg += key + ": {:.4f} - ".format(value)
            
        text = "\r{0}/{1}: [{2}] {3}Status: {4}".format(step, self.length, 
                                                         "="*(block-1) + arrow + \
                                                         "."*(barLength-block), 
                                                         data_msg, status)
        if len(text) < self.longest_len:
            diff = self.longest_len - len(text)
            text += ' '*diff
        elif len(text) > self.longest_len:
            self.longest_len = len(text)
            
        print(text, end='') # The \x1b[2K] is the escape sequence telling the print to erase the previous line.