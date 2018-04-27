import glob
import tifffile
import datetime
import os
from collections import abc
import numpy as np
from analysis import StackArray

class FLSImageDirSet(abc.Sequence):
    def __init__(self, directory):
        self._dns = sorted(glob.glob(directory + '/t*/'),
                          key = lambda x: int(os.path.split(os.path.split(x)[0])[1][1:]))
        with tifffile.TiffFile(self._dns[0] + 'actin_original.TIF', 'r') as tif:
            self._zero_timepoint = datetime.datetime.strptime(
                tif.metaseries_metadata['PlaneInfo']['acquisition-time-local'],
                '%Y%m%d %H:%M:%S.%f')
        self.nr_timepoints = len(self._dns)
        
    def __getitem__(self, timepoint):
        with tifffile.TiffFile(self._dns[timepoint] + 'actin_original.TIF', 'r') as tif:
            time = datetime.datetime.strptime(
                tif.metaseries_metadata['PlaneInfo']['acquisition-time-local'],
                '%Y%m%d %H:%M:%S.%f')
            confocal = StackArray(tif.asarray(), acquisition_time=time)
        return confocal
    
    def __len__(self):
        return self.nr_timepoints
