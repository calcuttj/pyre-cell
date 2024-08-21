import units, responses, HfFilter, ElecResponse, Node, torch, numpy as np

class decon_2d(Node.Node):
  #ignoring alternate side collection plane 
  def __init__(self, field_file, nwires=[800,800,480], nticks=6000):
    super().__init__()
    self.nwires = nwires
    self.nticks = nticks

    #Load field response from garfield file
    field_response = responses.load(field_file)
    self.period = field_response.period
    self.field_resp_samples = len(field_response.planes[0].paths[0].current)
    tensor_response = responses.wire_region_average(field_response).as_tensor()



    #do the FFT on avg response
    self.avg_response = [
        torch.fft.rfft(
          tensor_response[i], axis=1)
        for i, nw in enumerate(nwires)
    ]

    #Make an electronics response instance, defined by gain and shaping time
    ##Hardcoded -- make configurable
    self.elec_response = ElecResponse.ElecResponse(14., 2.)

    #Make an electronics response in the same coarseness of the field response
    #Units of period is ns
    self.elec_response_for_field = self.elec_response(
        torch.Tensor(torch.linspace(0, 1.e-3*self.period*self.field_resp_samples, self.field_resp_samples)))
    print(len(self.elec_response_for_field))

    ##Hardcoded -- make configurable
    self.scale_factor = ((1 << 12) - 1) / 1.4e3 #ADC / mV
    print(self.scale_factor)

    # Scale elec response, fft it, multiply the field response, inverse fft, and redigitze
    self.avg_response = [
      responses.redigitize(
        torch.fft.irfft(
          ar*torch.fft.rfft(self.elec_response_for_field * -1 * self.scale_factor)*self.period*1.e-3,
          axis=1
        ),
        self.period*1.e-3, .5, self.nticks
      ) for ar in self.avg_response
    ]

    paddings = [
      torch.zeros(nw, self.nticks) for nw in self.nwires
    ]

    for i, p in enumerate(paddings):
      p[:len(self.avg_response[i])] = self.avg_response[i]

    self.avg_response = [torch.fft.fft(torch.fft.rfft(ar, axis=1), axis=0) for ar in paddings]


    space = torch.zeros(480)
    space[:240] = torch.linspace(0, 1, 240)
    space[240:] = torch.linspace(1, 0, 240)
    self.filter = torch.exp(
      -.5*(space/np.sqrt(units.pi))**2
    )


  #Move wire and time axes up to init
  def forward(self, x, wire_axis=0, time_axis=1):
    ##FFT time view, then wire
    x = torch.fft.fft(torch.fft.rfft(x, axis=time_axis), axis=wire_axis)
    print(x.shape)

    x = x / self.avg_response[2]
    
    return (x.T * self.filter).T
