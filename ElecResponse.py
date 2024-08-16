import Node
import torch

class ElecResponse(Node.Node):
  def __init__(self, gain, shaping):
    super().__init__()
    self.gain = gain
    self.shaping = shaping

  def response(self, time):
    ## Fix this
    #if (time <= 0 || time >= 10 * units::microsecond) {  #range of validity
     #   return 0.0;
    #}

    ##shortcutting
    gain = self.gain
    shaping = self.shaping

    reltime = time / shaping

    #a scaling is needed to make the anti-Lapalace peak match the expected gain
    #fixme: this scaling is slightly dependent on shaping time.  See response.py
    gain *= 10 * 1.012

    from torch import exp, cos, sin
    return (4.31054 * exp(-2.94809 * reltime) * gain - 2.6202 * exp(-2.82833 * reltime) * cos(1.19361 * reltime) * gain -
            2.6202 * exp(-2.82833 * reltime) * cos(1.19361 * reltime) * cos(2.38722 * reltime) * gain +
            0.464924 * exp(-2.40318 * reltime) * cos(2.5928 * reltime) * gain +
            0.464924 * exp(-2.40318 * reltime) * cos(2.5928 * reltime) * cos(5.18561 * reltime) * gain +
            0.762456 * exp(-2.82833 * reltime) * sin(1.19361 * reltime) * gain -
            0.762456 * exp(-2.82833 * reltime) * cos(2.38722 * reltime) * sin(1.19361 * reltime) * gain +
            0.762456 * exp(-2.82833 * reltime) * cos(1.19361 * reltime) * sin(2.38722 * reltime) * gain -
            2.620200 * exp(-2.82833 * reltime) * sin(1.19361 * reltime) * sin(2.38722 * reltime) * gain -
            0.327684 * exp(-2.40318 * reltime) * sin(2.5928 * reltime) * gain +
            +0.327684 * exp(-2.40318 * reltime) * cos(5.18561 * reltime) * sin(2.5928 * reltime) * gain -
            0.327684 * exp(-2.40318 * reltime) * cos(2.5928 * reltime) * sin(5.18561 * reltime) * gain +
            0.464924 * exp(-2.40318 * reltime) * sin(2.5928 * reltime) * sin(5.18561 * reltime) * gain)

  def forward(self, x):
    return self.response(x)
