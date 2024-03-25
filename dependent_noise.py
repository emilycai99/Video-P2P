import enum
import math
import torch
import torch.nn as nn
import numpy as np

def toeplitz(c, r):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)

def construct_cov_mat(num_frames,decay_rate):
    seq = torch.pow(decay_rate,torch.arange(num_frames))
    return toeplitz(seq,seq)

def construct_ar_cov_mat(window_size,decay_rate,ar_coeff,num_window):
    seq = torch.pow(decay_rate,torch.arange(window_size))
    seq_c = torch.pow(torch.sqrt(torch.tensor(ar_coeff)),torch.arange(num_window))
    return torch.kron(toeplitz(seq_c,seq_c),toeplitz(seq,seq))

class dependent_noise_sampler(nn.Module):
    def __init__(self,         
            num_frames=60,
            decay_rate=0.1,
            window_size=60,
            ar_sample=False,
            ar_coeff=0.1,
            loss_sig=False
    ):
        super(dependent_noise_sampler, self).__init__()
        '''
        1. ar_sample = True + window_size < num_frames ---> dependent window design
        2. [ar_sample = True + window_size = num_frames] == [ar_sample = False + window_size = num_frames] --> no window design
        3. [ar_sample = False + window size < num_frames] --> independent window design
        4. what is loss_sig for? since ar_cov_mat is not used
        '''
        
        #------------------- new ----------------
        self.cov_mat = construct_cov_mat(window_size,decay_rate)
        self.cov_mat_inv = torch.inverse(self.cov_mat)
        # self.inv_cov_mat = torch.inverse(self.cov_mat)
        self.sampler = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(window_size),covariance_matrix=self.cov_mat)
        self.window_size = window_size
        self.window_num = int(num_frames / window_size)
        self.ar_sample = ar_sample
        self.ar_coeff = ar_coeff
        self.loss_sig = loss_sig
        if ar_sample and loss_sig:
            # I think this is for window design
            self.ar_cov_mat = construct_ar_cov_mat(window_size,decay_rate,ar_coeff,self.window_num)
            self.ar_cov_mat_inv = torch.inverse(self.ar_cov_mat)
    
    def sample(self, input):
        # x: bsz x channel x frame x height x width
        x = input.permute(0, 1, 3, 4, 2)
        # print('x.shape', x.shape)
        
        if self.ar_sample:
            noise = torch.zeros(x.shape)
            for i in range(self.window_num):
                '''
                1. sampler will return a variable of length: window size
                2. we need to specify where is the batch size and treat everything else as a single dimension
                '''
                if i == 0:
                    noise[:, :, :, :, i*self.window_size:(i+1)*self.window_size] = self.sampler.sample(x.shape[:-1])
                else:
                    noise[:, :, :, :, i*self.window_size:(i+1)*self.window_size] = math.sqrt(self.ar_coeff) * noise[:, :, :, :, (i-1)*self.window_size:i*self.window_size] + math.sqrt(1-self.ar_coeff)  * self.sampler.sample(x.shape[:-1])
                    # noise[:,i*self.window_size:(i+1)*self.window_size,:] = math.sqrt(self.ar_coeff) * noise[:,(i-1)*self.window_size:i*self.window_size,:] + math.sqrt(1-self.ar_coeff)  * self.sampler.sample((x.shape[0],x.shape[2]))
            noise = noise.to(x.device)
        else:
            noise = torch.cat([self.sampler.sample(x.shape[:-1]) for i in range(self.window_num)],axis=-1).to(x.device)
        
        noise = torch.permute(noise,(0, 1, 4, 2, 3))
        noise = noise.to(x.dtype)
        assert noise.shape == input.shape, 'noise and input do not have the same shape'
        
        return noise

if __name__ == '__main__':
    # bsz x channel x frame x height x width
    x = torch.randn(size=(1, 4, 8, 3, 3))
    ar = dependent_noise(num_frames=8, ar_sample=False, loss_sig=True, window_size=8)
    noise = ar.sample(x)
    print(noise.shape)
    print(noise[0, 0, :, 0, 0])
    
    