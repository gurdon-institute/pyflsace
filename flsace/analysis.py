import numpy as np
import pandas as pd

from skimage.filters import gaussian
from skimage.filters import threshold_triangle
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import binary_opening, binary_dilation
from skimage.morphology.extrema import local_maxima

from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

import lapjv

class Stack(object):
    def __init__(self, confocal, base_slice,
                 tirfs={},
                 voxel_width=0.1487,
                 voxel_depth=1.,
                 sigma=1.5,
                 K=3,
                 thresholding=threshold_triangle,
                 link_cutoff_distance=2.,
                 progress=None, debug_link=False):
        self._confocal = confocal[base_slice:]
        self._tirfs = tirfs
        self._sigma = sigma
        self._K = K
        self._thresholding = thresholding
        
        self._voxel_width = voxel_width
        self._voxel_depth = voxel_depth
        
        self._link_cutoff_distance = link_cutoff_distance
        
        if progress is None:
            self._progress = lambda x, *args, **kw: x
        else:
            self._progress = progress

        self._generate_rois()
        if not debug_link:
            self._link_slices()

    def _generate_rois(self):
        slice_rois = []
        masks = []
        for i in self._progress(range(self._confocal.shape[0]), desc="Generating ROIs"):
            dog = (gaussian(self._confocal[i], sigma=self._sigma) -
                   gaussian(self._confocal[i], sigma=self._K*self._sigma))
            thresh = self._thresholding(dog)
            binary = dog > thresh
            distance = gaussian(ndi.distance_transform_edt(binary), sigma=self._sigma)
            local_max = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
            #local_max = local_maxima(distance, selem=np.ones((9, 9)))
            markers = ndi.label(local_max)[0]
            labels = watershed(-distance, markers, mask=binary)
            labels *= binary_opening(labels > 0)
            masks.append(labels > 0)
            slice_rois.append(regionprops(labels))
        self._roi_masks = masks
        self._slice_rois = slice_rois
        
    def _link_slices(self):
        vw = self._voxel_width
        vd = self._voxel_depth
        fls = [[r] for r in self._slice_rois[0]]
        for i in self._progress(range(1, len(self._slice_rois)), desc="Linking slices"):
            next_rois = self._slice_rois[i]
            valid_fls = fls #[f for f in fls if len(f) >= i-self._link_]
            if len(valid_fls) == 0 or len(next_rois) == 0:
                break
            X = vw*np.array([f[-1].centroid for f in valid_fls])
            Y = vw*np.array([r.centroid for r in next_rois])
            C = pairwise_distances(X, Y)
            eC = (1.1*self._link_cutoff_distance)*np.ones((max(C.shape), max(C.shape)))
            eC[:C.shape[0], :C.shape[1]] = C
            C = eC
            Yidx, _, _ = lapjv.lapjv(C)
            for xi, yi in enumerate(Yidx):
                if C[xi, yi] < self._link_cutoff_distance:
                    valid_fls[xi].append(next_rois[yi])

#             Xidx, Yidx = linear_sum_assignment(C)
#             for xi, yi in zip(Xidx, Yidx):
#                 if C[xi, yi] < self._link_cutoff_distance:
#                     valid_fls[xi].append(next_rois[yi])
        self._fls = fls
        
    @property
    def path_lengths(self):
        return [np.linalg.norm(np.diff([list(np.array(r.centroid)*self._voxel_width) + [i*self._voxel_depth]
                                        for i, r in enumerate(f)], axis=0), axis=1).sum()
                for f in self._fls]
    
    def _fls_to_dict(self, rs, do_shaft_outline=True, do_intensities=True):
        vw = self._voxel_width
        vd = self._voxel_depth
        straight = ((((np.array(rs[-1].centroid)-np.array(rs[0].centroid))*vw)**2).sum() +
                    ((len(rs)-1)*vd)**2)**0.5
        path = np.linalg.norm(np.diff(
            [[r.centroid[0]*vw, r.centroid[1]*vw, i*vd]
             for i, r in enumerate(rs)],
            axis=0), axis=1).sum()
                
        fls_properties = {
            "straight_length": straight,
            "path_length": path,
            "base_area": rs[0].area*(vw**2),
            "straightness": straight / path if straight > 0 else float("NaN"),
            "circularity": 4*np.pi*rs[0].area/rs[0].perimeter**2 if rs[0].perimeter > 0 else float("NaN"),
        }
            
        shaft_area = [roi.area*(vw**2) for roi in rs]
        fls_properties['shaft_area'] = np.array(shaft_area)

        shaft_coordinates = [[roi.centroid[0]*vw, roi.centroid[1]*vw, i*vd]
                             for i, roi in enumerate(rs)]
        fls_properties['shaft_coordinates'] = np.array(shaft_coordinates)

        if do_shaft_outline:
            shaft_outline = []
            for i, roi in enumerate(rs):
                base_indices = roi.coords
                shaft_coordinates.append(list(self._voxel_width*np.array(roi.centroid)) + [i*self._voxel_depth])
                try:
                    hull = base_indices[ConvexHull(base_indices).vertices]
                    shaft_outline.append(self._voxel_width*np.array(hull))
                except QhullError:
                    shaft_outline.append(self._voxel_width*np.array(base_indices))
            fls_properties['shaft_outline'] = shaft_outline

        
        if do_intensities:
            base_indices = rs[0].coords
            foreground_mask = np.zeros_like(self._confocal[0], dtype=bool)
            x, y = base_indices[:, 0], base_indices[:, 1]
            foreground_mask[x, y] = True

            bounding_box = foreground_mask[x.min():x.max()+1, y.min():y.max()+1]
            dil1 = binary_dilation(foreground_mask, bounding_box)
            dil2 = binary_dilation(dil1, bounding_box)
            background_mask = dil2 & (~dil1) & (~self._roi_masks[0])
            nr_bg_pxls = background_mask.sum()

            confocal_mean_intensity = self._confocal[0][foreground_mask].mean()
            if nr_bg_pxls > 10:
                confocal_mean_background_intensity = self._confocal[0][background_mask].mean()
                confocal_std_background_intensity = self._confocal[0][background_mask].std()
            else:
                confocal_mean_background_intensity = float("NaN")
                confocal_std_background_intensity = float("NaN")
            fls_properties["confocal_mean_intensity"] = confocal_mean_intensity
            fls_properties["confocal_mean_background_intensity"] = confocal_mean_background_intensity

        
            for tirf_name, tirf_img in self._tirfs.items():
                fls_properties[tirf_name+"_mean_intensity"] = tirf_img[foreground_mask].mean()
                if nr_bg_pxls > 10:
                    fls_properties[tirf_name+"_mean_background_intensity"] = tirf_img[background_mask].mean()
                    fls_properties[tirf_name+"_std_background_intensity"] = tirf_img[background_mask].std()
                else:
                    fls_properties[tirf_name+"_mean_background_intensity"] = float("NaN")
                    fls_properties[tirf_name+"_std_background_intensity"] = float("NaN")
            
            shaft_mean_intensity = []
            shaft_mean_background_intensity = []
            shaft_std_background_intensity = []
        
            for i, roi in enumerate(rs):
                base_indices = roi.coords
                foreground_mask = np.zeros_like(self._confocal[i], dtype=bool)
                x, y = base_indices[:, 0], base_indices[:, 1]
                foreground_mask[x, y] = True

                bounding_box = foreground_mask[x.min():x.max()+1, y.min():y.max()+1]
                dil1 = binary_dilation(foreground_mask, bounding_box)
                dil2 = binary_dilation(dil1, bounding_box)
                background_mask = dil2 & (~dil1) & (~self._roi_masks[i])
                nr_bg_pxls = background_mask.sum()

                shaft_mean_intensity.append(self._confocal[i][foreground_mask].mean())
                if nr_bg_pxls > 10:
                    shaft_mean_background_intensity.append(self._confocal[i][background_mask].mean())
                    shaft_std_background_intensity.append(self._confocal[i][background_mask].std())
                else:
                    shaft_mean_background_intensity.append(float("NaN"))
                    shaft_std_background_intensity.append(float("NaN"))
                
            fls_properties['shaft_mean_intensity'] = np.array(shaft_mean_intensity)
            fls_properties['shaft_mean_background_intensity'] = np.array(shaft_mean_background_intensity)
            fls_properties['shaft_std_background_intensity'] = np.array(shaft_std_background_intensity)
        
        return fls_properties
    
    def get_table(self, *a, **kw):
        return pd.DataFrame([self._fls_to_dict(f, *a, **kw)
                             for f in self._progress(self._fls, desc='Table Output')])

class FLS:
    def __init__(self, frame, row):
        self.frames = []
        self.path_length = []
        self.base_area = []
        self.base_position = []
        
        self.add_row(frame, row)
    
    def add_row(self, frame, row):
        self.frames.append(frame)
        self.path_length.append(row.path_length)
        self.base_area.append(row.base_area)
        self.base_position.append(np.array(row.shaft_coordinates[0]))

class Frames:
    def __init__(self, confocals, base_slice,
                 voxel_width=0.1487,
                 voxel_depth=1.,
                 sigma=1.5,
                 K=3,
                 thresholding=threshold_triangle,
                 link_cutoff_distance=2.,
                 progress=None):
        self.progress = progress
        
        self.stack_tables = [
            (confocal.acquisition_time,
             Stack(confocal,
                  base_slice,
                  voxel_width=voxel_width,
                  voxel_depth=voxel_depth,
                  sigma=sigma,
                  K=K,
                  thresholding=thresholding,
                  link_cutoff_distance=link_cutoff_distance,
                  progress=progress).get_table())
            for confocal in progress(confocals, desc='Segmenting')]
        self.frame_times = [time for time, _ in self.stacks]
        
        self._link_frames()
        
    def _link_frames(self):
        def cost_matrix(flss, cands, frame, cost_thresh=1e3):
            # Relevant FLS are from the last 4 frames
            rel_flss = [fls for fls in flss if fls.frames[-1] >= frame-4]

            # Position arrays for existing and candidate FLS
            flss_pos = np.array([np.mean(fls.base_position, axis=0) for fls in rel_flss])
            cand_pos = np.array([row.shaft_coordinates[0] for _, row in cands.iterrows()])

            # Calculate pairwise distances as cost
            fp0, cp0 = np.meshgrid(flss_pos[:, 0], cand_pos[:, 0])
            fp1, cp1 = np.meshgrid(flss_pos[:, 1], cand_pos[:, 1])
            # And fill into cost matrix
            C = ((fp0-cp0)**2+(fp1-cp1)**2)
            
            # Make cost matrix square
            eC = cost_thresh+np.random.rand(max(C.shape), max(C.shape))*cost_thresh
            eC[:C.shape[0], :C.shape[1]] = C
            return rel_flss, eC
        
        # Initial FLS collection is everything in the first frame
        flss = [FLS(0, row) for i, row in self.stack_tables[0].iterrows()]
        # Iterate over tables from time points, starting at the second one
        for i in self.progress(range(1, len(self.stack_tables)), desc='Framelink'):
            tab = self.stack_tables[i][0]
            # Calculate cost matrix
            rel_flss, C = cost_matrix(flss, tab, i)
            # Find best assignments
            col_ind, row_ind, _ = lapjv.lapjv(C)
            nr_fls = len(rel_flss)
            nr_cand = tab.shape[0]
            for r in range(nr_cand):
                c = col_ind[r]
                cost = C[r, c]
                if c >= nr_fls or cost > 1.:
                    flss.append(FLS(i, tab.iloc[r]))
                else:
                    rel_flss[c].add_row(i, tab.iloc[r])
        self.flss = flss
        
    def get_table(self):
        rows = []
        fls_id = 0
        for f in self.flss:
            for i, frame in enumerate(f.frames):
                rows.append((fls_id, frame, self.frame_times[frame],
                             f.path_length[i],
                             f.base_area[i],
                             f.base_position[i][0],
                             f.base_position[i][1]))
        return pd.DataFrame(data=rows,
                            columns=["fls_index", "frame", "time",
                                     "path_length", "base_area", "X", "Y"])
            
if __name__ == '__main__':
    import NDParser
    from tqdm import tqdm
    nd = NDParser.NDFile('../test_imgs/control.nd')
    fa = FLSAce(nd.get_image_data('Confocal 560', 1, 1), 1,
                tirfs={
                    "NWASP": nd.get_image_data(2, 1, 1),
                    "TIRFactin": nd.get_image_data(4, 1, 1),
                    "ena": nd.get_image_data(6, 1, 1),
                }, progress=tqdm)
    df = fa.get_table()
    print(df.shaft_outline[1])
