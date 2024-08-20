import units, responses, HfFilter, ElecResponse, Node, torch

class decon_2d(Node.Node):
  def __init__(self, field_file):
    super().__init__()

    #Load field response from garfield file
    self.field_response = responses.load(field_file)
    self.period = self.field_response.period
    self.field_resp_samples = len(self.field_response.planes[0].paths[0].current)
    self.field_response = self.field_response.as_tensor()

    #Make an electronics response instance, defined by gain and shaping time
    ##Hardcoded -- make configurable
    self.elec_response = ElecResponse.ElecResponse(14., 2.*units.us)

    #Make an electronics response in the same coarseness of the field response
    #Units of period is ns
    self.elec_response_for_field = self.elec_response(
        torch.Tensor(torch.linspace(0, self.period*self.field_resp_samples, self.field_resp_samples)))

    ##Hardcoded -- make configurable
    self.scale_factor = ((1 << 12) - 1) / 1.4e3 #ADC / mV

  def forward(self, x, wire_axis=0, time_axis=1):
    ##FFT time view
    x = torch.fft.fft(x, axis=time_axis)
    ##FFT wire view
    x = torch.fft.fft(x, axis=wire_axis)

    the_response = responses.redigitize(self.elec_response_for_field*self.field_response[2], self.period*1.e-3, .5, 6000)

    x /= torch.fft.fft(torch.fft.fft(the_response, axis=time_axis), wire_axis)
    
    filter_vals = torch.zeros(480)
    filter_vals[:240] = torch.linspace(0, 1, 240)
    filter_vals[240:] = torch.linspace(0, 1, 240)[::-1]
    x = x.T * torch.exp(
      .5*(filter_vals/torch.sqrt(units.pi))
    ).T
    return x
