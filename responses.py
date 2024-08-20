import torch, json
import units

class PathResponse:
  def __init__(self, current, pitchpos, wirepos):
    #super().__init__()
    self.current = torch.Tensor(current)
    self.pitchpos = pitchpos
    self.wirepos = wirepos

class PlaneResponse:
  def __init__(self, paths, planeid, location, pitch):
    #super().__init__()
    self.paths = paths 
    self.planeid = planeid
    self.location = location 
    self.pitch = pitch 

class FieldResponse:
  def __init__(self, planes, axis, origin, tstart, period, speed):
    #super().__init__()
    self.planes = planes
    self.axis = axis
    self.origin = origin
    self.tstart = tstart
    self.period = period
    self.speed = speed

  def as_tensor(self):
    results = torch.zeros(len(self.planes),
                          len(self.planes[0].paths),
                          self.planes[0].paths[0].current.shape[0])
    for i, plane in enumerate(self.planes):
      for j, path in enumerate(plane.paths):
        results[i, j] = path.current 
    return results

def load(fname):
  import bz2
  with bz2.open(fname) as f:
    return load_field(json.load(f)['FieldResponse'])

def load_plane(plane):
    paths = [load_path(p['PathResponse']) for p in plane['paths']]
    return PlaneResponse(
        paths, plane['planeid'], plane['location'], plane['pitch'])

def load_path(path):
    return PathResponse(
        path['current']['array']['elements'], path['pitchpos'], path['wirepos'])
    

def load_field(field):
    planes = [load_plane(p['PlaneResponse']) for p in field['planes']]
    return FieldResponse(
        planes,
	field['axis'],
	field['origin'],
	field['tstart'],
	units.ns*field['period'],
	field['speed'])

from collections import defaultdict
import numpy as np

#This can be optimized
def wire_region_average(fr):
    newplanes = []
    
    #Loop over planes
    for plane in fr.planes:
        newpaths = []
        pitch = plane.pitch

        #Default values
        avgs = defaultdict(lambda: torch.Tensor(np.zeros(len(plane.paths[0].current))))
        fresp_map = {}
        pitch_pos_range_map = {}

        nsamples = 0

        #Loop over the paths for this plane  
        for path in plane.paths:

          #Get the index for this sub-pitch path
          eff_num = int(path.pitchpos / (0.01 * pitch))

          #+- the sub-pitch
          for en in [eff_num, -eff_num]:

            if en not in fresp_map:
                #If this is not in the map, make a new entry 
                fresp_map[en] = path.current
            else:
                ##If in the map, average it out
                fresp_map[en] += path.current
                fresp_map[en] *= .5

        pitch_pos = list(fresp_map.keys())
        pitch_pos.sort() ##Do this?

        #loop over positions within the wire pitch, and set the lower/upper
        #bounds for that position
        min_val = -1e9
        max_val = 1e9
        for i, pos in enumerate(pitch_pos):
            if i == 0:
                pitch_pos_range_map[pos] = (
                    min_val, 
                    (pitch_pos[i] + pitch_pos[i + 1]) / 2. * 0.01 * pitch
                )
            elif i == len(pitch_pos) - 1:
                pitch_pos_range_map[pos] = (
                    (pitch_pos[i] + pitch_pos[i - 1]) / 2. * 0.01 * pitch,
                    max_val
                )
            else:
                pitch_pos_range_map[pos] = (
                    (pitch_pos[i] + pitch_pos[i - 1]) / 2. * 0.01 * pitch,
                    (pitch_pos[i] + pitch_pos[i + 1]) / 2. * 0.01 * pitch
                )

        wire_regions = set()
        for pos in pitch_pos:
            if pos > 0:
                wire_regions.add(
                    round((pos * 0.01 * pitch - 0.001 * pitch) / pitch)
                )
            else:
                wire_regions.add(
                    round((pos * 0.01 * pitch + 0.001 * pitch) / pitch)
                )

        for wire_no in wire_regions:
            #avgs[wire_no] = torch.Tensor(np.zeros_like())
            for resp_num, response in fresp_map.items():
                low_limit, high_limit = pitch_pos_range_map[resp_num]
                low_limit = max(low_limit, (wire_no - 0.5) * pitch)
                high_limit = min(high_limit, (wire_no + 0.5) * pitch)

                if high_limit > low_limit:
                    avgs[wire_no] += response * (high_limit - low_limit) / pitch

        for region, response in avgs.items():
            newpaths.append(
                PathResponse(response.tolist(), region * pitch, 0.0)
            )

        newplanes.append(
            PlaneResponse(newpaths, plane.planeid, plane.location, plane.pitch)
        )

    return FieldResponse(newplanes, fr.axis, fr.origin, fr.tstart, fr.period, fr.speed)


def redigitize(x, avg_period, target_period, target_ticks):
  # redigitize ...
  source_ticks = len(x[0])
  results = torch.zeros(x.shape[0], target_ticks)
  fcount = 1;
  for i in range(target_ticks):
    target_time = i*target_period

    if (fcount < source_ticks):
      while (target_time > fcount*avg_period and fcount < source_ticks):
        fcount += 1
        #if (fcount >= source_ticks) break;

    if (fcount < source_ticks):
      results[:, i] = ((target_time - avg_period*(fcount-1)) / avg_period * x[:, fcount - 1] +
                   (avg_period*(fcount) - target_time) / avg_period * x[:, fcount])
    else:
      results[:, i] = 0;

  return results
