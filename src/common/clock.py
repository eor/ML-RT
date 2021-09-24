import torch
import numpy as np


class Clock:
    def __init__(self, itr_warmup=10, itr_average_time=20):
        self.itr_warmup = itr_warmup
        self.itr_average_time = itr_average_time
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.device = 'cuda'

    def get_time(self, model, positional_arguements):

        # ENSURE INPPUT IS ON RIGHT DEVICE
        model.to(self.device)
        positional_arguements = [arg.to(self.device)
                                 for arg in positional_arguements]

        time_keeping_array = np.zeros((self.itr_average_time, 1))

        # GPU WARM_UP
        for _ in range(self.itr_warmup):
            _ = model(*positional_arguements)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for i in range(self.itr_average_time):
                self.starter.record()
                _ = model(*positional_arguements)
                self.ender.record()
                # WAIT FOR GPU SYNCHRONIZATION
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender)
                time_keeping_array[i] = curr_time

        avg_time = np.sum(time_keeping_array) / self.itr_average_time
        std_time = np.std(time_keeping_array)

        return avg_time, std_time
