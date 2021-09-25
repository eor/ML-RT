import torch
import numpy as np

""" 
Class to measure inference time of a model on GPU
(Note: Ensure that all models and input parameters are already on
GPU, before passing to this class.)
"""
class Clock:
    def __init__(self, itr_warmup=20, itr_average_time=100):
        self.itr_warmup = itr_warmup
        self.itr_average_time = itr_average_time
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.device = 'cuda'

    def get_time(self, function, positional_arguements):
        
        print('\n \tMeasuring Inference time. Please wait. \n')
        
        time_keeping_array = np.zeros((self.itr_average_time, 1))

        # GPU WARM_UP
        for _ in range(self.itr_warmup):
            _ = function(*positional_arguements)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for i in range(self.itr_average_time):
                self.starter.record()
                _ = function(*positional_arguements)
                self.ender.record()
                
                # WAIT FOR GPU SYNCHRONIZATION
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender)
                time_keeping_array[i] = curr_time

        avg_time = np.sum(time_keeping_array) / self.itr_average_time
        std_time = np.std(time_keeping_array)

        return avg_time, std_time
