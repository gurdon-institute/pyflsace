import glob
import tifffile
import datetime
import os
from collections import abc
import numpy as np

class StackArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, acquisition_time=None):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)                                 
        obj.acquisition_time = acquisition_time
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.acquisition_time = getattr(obj, 'acqisition_time', None)

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