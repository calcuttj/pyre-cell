import Node
import numpy as np
import torch

class HfFilter(Node.Node):
  def __init__(self, max_freq=1, sigma=.12):
    super().__init__()
    self.max_freq = max_freq
    #self.sig = 10/np.sqrt(np.pi)
    self.sigma = sigma 

  def forward(self, x, dim):
    ##Assumes evenness oh well
    #dim = 1
    #print(x)
    #print(x.shape)
    the_filter = torch.zeros(x.shape[dim])
    #print(the_filter.shape)

    freqs = np.linspace(0., self.max_freq, (x.shape[dim]-1))
    #print(freqs)
    the_filter[1:] = torch.tensor(freqs)
    #print(the_filter)
    the_filter = torch.exp(-.5*(the_filter/self.sigma)**2)
    #print(the_filter)
    the_filter[0] = 0.
    return x * the_filter
