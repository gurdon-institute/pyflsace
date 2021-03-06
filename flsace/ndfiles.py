import tifffile
import os
try:
    from .analysis import StackArray
except SystemError:
    from analysis import StackArray
from collections import abc
import datetime

def _strip_quotes(s):
    s = s.strip()
    if s[0] == '"':
        s = s[1:]
    if s[-1] == '"':
        s = s[:-1]
    return s

class NDFile(object):
    def __init__(self, filename):
        self._filename = filename
        with open(filename, 'r') as f:
            lines = f.readlines()

        self._keyvals = {}
        for line in lines:
            parts = line.split(', ')
            key = _strip_quotes(parts[0])
            if len(parts) > 2:
                value = parts[1:]
            elif len(parts) == 1:
                continue
            else:
                value = _strip_quotes(parts[1])
            if value == "TRUE":
                value = True
            elif value == "FALSE":
                value = False

            if key == 'WavePointsCollected':
                wave_id = value[0].strip()
                value = [int(v.strip()) for v in value[1:]]
                key += wave_id
            self._keyvals[key] = value

    @property
    def number_of_wavelengths(self):
        if "NWavelengths" not in self._keyvals:
            return 1
        if not self._keyvals["DoWave"]:
            return 1
        return int(self._keyvals["NWavelengths"])

    @property
    def filename_prefix(self):
        return os.path.splitext(os.path.split(self._filename)[1])[0]
    
    @property
    def path(self):
        return os.path.split(self._filename)[0]
    
    @property
    def wavelengths(self):
        if not self._keyvals["DoWave"]:
            return [str(1)]
        return [self._keyvals["WaveName%d" % (i+1)]
                for i in range(self.number_of_wavelengths)]

    @property
    def number_of_stage_positions(self):
        if "NStagePositions" not in self._keyvals:
            return 1
        return int(self._keyvals["NStagePositions"])
    
    @property
    def number_of_time_points(self):
        if "NTimePoints" not in self._keyvals:
            return 1
        return int(self._keyvals["NTimePoints"])

    def __repr__(self):
        return "\n".join(["%s: %s" % (key, str(value)) for
                          key, value in self._keyvals.items()])

    def get_image_filename(self, wavelength, stage_position, time_point):
        if type(wavelength) == str:
            wavelength_name = wavelength
            try:
                wavelength_number = self.wavelengths.index(wavelength)+1
            except IndexError:
                raise ValueError("Invalid wavelength name")
        else: # assume it is an int
            if wavelength < 1 or wavelength > self.number_of_wavelengths:
                raise ValueError("Invalid wavelength number")
            wavelength_name = self.wavelengths[wavelength-1]
            wavelength_number = wavelength

        if stage_position < 1 or stage_position > self.number_of_stage_positions:
            raise ValueError("Invalid stage position")

        if time_point < 1 or time_point > self.number_of_time_points:
            raise ValueError("Invalid time point")

        # Assemble file name
        parts = [self.path + "/" + self.filename_prefix]
        if self._keyvals["DoWave"]:
            if self._keyvals["WaveInFileName"]:
                parts.append("w%d%s" % (wavelength_number, wavelength_name))
            else:
                parts.append("w%d" % wavelength_number)
        if self._keyvals["DoStage"]:
            parts.append("s%d" % stage_position)
        if self._keyvals["DoTimelapse"]:
            parts.append("t%d" % time_point)

        return "_".join(parts) + ".TIF"

    def all_images(self):
        imgs = {}
        for w in range(1, self.number_of_wavelengths+1):
            for s in range(1, self.number_of_stage_positions+1):
                for t in range(1, self.number_of_time_points+1):
                    if 'WavePointsCollected' + str(w) in self._keyvals:
                        if t not in self._keyvals['WavePointsCollected' + str(w)]:
                            continue
                    imgs[self.get_image_filename(w, s, t)] = {
                        'wavelength_name': self.wavelengths[w-1],
                        'wavelength_number': w,
                        'stage_position': s,
                        'time_point': t,
                    }
        return imgs

    def get_images_for_stage_and_timepoint(self, stage_position, time_point):
        if stage_position < 1 or stage_position > self.number_of_stage_positions:
            raise ValueError("Invalid stage position")

        if time_point < 1 or time_point > self.number_of_time_points:
            raise ValueError("Invalid time point")

        imgs = {}
        for w in range(1, self.number_of_wavelengths+1):
            if 'WavePointsCollected' + str(w) in self._keyvals:
                if time_point not in self._keyvals['WavePointsCollected' + str(w)]:
                    continue
            imgs[(w, self.wavelengths[w-1])] = self.get_image_filename(w, stage_position, time_point)
        return imgs

    def get_image_data(self, wavelength, stage_position, time_point):
        with tifffile.TiffFile(
                self.get_image_filename(wavelength,
                                        stage_position,
                                        time_point), 'r') as tif:
            data = tif.asarray()
        return data
    
class ImageSet(abc.Sequence):
    def __init__(self, ndfilename,
                 confocal_wavelength=1,
                 stage_position=1,
                 rename_wavelengths=lambda x: x):
        self._ndfile = NDFile(ndfilename)
        self._confocal_wavelength = confocal_wavelength
        self._stage_position = stage_position
        self.nr_timepoints = self._ndfile.number_of_time_points
        self._renamer = rename_wavelengths

    def __getitem__(self, timepoint):
        if timepoint < 0 or timepoint >= self._ndfile.number_of_time_points:
            raise IndexError
        imgs = self._ndfile.get_images_for_stage_and_timepoint(self._stage_position, timepoint+1)

        images = {}
        for (wnum, wname), filename in imgs.items():
            name = self._renamer(wname)
            with tifffile.TiffFile(filename, 'r') as tif:
                time = datetime.datetime.strptime(
                    tif.metaseries_metadata['PlaneInfo']['acquisition-time-local'],
                    '%Y%m%d %H:%M:%S.%f')
                images[name] = StackArray(tif.asarray(), acquisition_time=time)
        return images

    def __len__(self):
        return self.nr_timepoints

if __name__ == "__main__":
    import sys
    from pprint import pprint

    names = {'Confocal 470': 'confocal',
             'TIRF561+Quad': 'VASP',
             'TIRF640+Quad': 'Ena'}
    
    imgs = ImageSet((sys.argv[1]), rename_wavelengths=lambda x: names[x])
    print(imgs[1])
