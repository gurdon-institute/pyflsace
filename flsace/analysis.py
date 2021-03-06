import numpy as np
import pandas as pd

from skimage.filters import gaussian
from skimage.filters import threshold_triangle
from skimage.segmentation import watershed
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

class StackArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, acquisition_time=None):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)                                 
        obj.acquisition_time = acquisition_time
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.acquisition_time = getattr(obj, 'acqisition_time', None)

def _pxlline(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    delta = y - x
    t = np.linspace(0, 1, int(np.ceil(np.linalg.norm(x-y))+1))
    return np.unique((x + t[:, np.newaxis]*delta).astype(int), axis=0).T

class Stack(object):
    def __init__(self, confocal, base_slice,
                 tirfs={},
                 voxel_width=0.1487,
                 voxel_depth=1.,
                 sigma=1.5,
                 K=3,
                 thresholding=threshold_triangle,
                 link_cutoff_distance=2.,
                 alternative_linking=False,
                 progress=None, progress_offset=None):
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
        self._progress_offset = progress_offset

        self._generate_rois()

        if alternative_linking:
            self._alt_link_slices()
        else:
            self._link_slices()

    def _generate_rois(self):
        slice_rois = []
        masks = []
        dog = (gaussian(self._confocal, sigma=(0, self._sigma, self._sigma)) -
               gaussian(self._confocal, sigma=(0, self._K*self._sigma, self._K*self._sigma)))
        thresh = self._thresholding(dog)
        binary = dog > thresh
        for i in self._progress(range(self._confocal.shape[0]), desc="Generating ROIs",
                                position=self._progress_offset):
            distance = gaussian(ndi.distance_transform_edt(binary[i]), sigma=self._sigma)
            local_max = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary[i])
            #local_max = local_maxima(distance, selem=np.ones((9, 9)))
            markers = ndi.label(local_max)[0]
            labels = watershed(-distance, markers, mask=binary[i])
            labels *= binary_opening(labels > 0)
            masks.append(labels > 0)
            slice_rois.append(regionprops(labels))
        self._roi_masks = masks
        self._slice_rois = slice_rois
        
    def _alt_link_slices(self, cutoff=1.5):
        vw = self._voxel_width
        vd = self._voxel_depth
        conv = np.array([vw, vw, vd])
        intens = gaussian(self._confocal, self._sigma)
        maxv = 512*intens.max()
        fls = [[(0, r, np.array(list(r.centroid) + [0]))]
               for r in self._slice_rois[0]]
        pool = [(z, r, np.array(list(r.centroid) + [z]))
                for z in range(1, len(self._slice_rois))
                for r in self._slice_rois[z]]
        matched = True
        p = self._progress(desc="Linking slices", position=self._progress_offset)
        while matched and (len(pool) > 0) and (len(fls) > 0):
            X = np.array([f[-1][2] for f in fls])
            Y = np.array([p[2] for p in pool])
            P = pairwise_distances(X*conv, Y*conv)
            C = maxv*np.ones((X.shape[0], Y.shape[0]))
            ci, cj = np.where(P < cutoff)
            for i, j in zip(ci, cj):
                x, y, z = _pxlline(X[i], Y[j])
                C[i, j] = intens[z, x, y].mean()/P[i, j]
            xm, ym = np.unravel_index(np.argsort(-C, axis=None), C.shape)
            candidates = P[xm, ym] < cutoff
            xm = xm[candidates]
            ym = ym[candidates]
            looked_at_x = []
            looked_at_y = []
            matched = False
            for xi, yi in zip(xm, ym):
                if xi not in looked_at_x and yi not in looked_at_y:
                    matched = True
                    looked_at_x.append(xi)
                    looked_at_y.append(yi)
                    fls[xi].append(pool[yi])
            pool = [pool[yi] for yi in range(len(pool)) if yi not in looked_at_y]
            p.update(1)
        p.close()
        self._fls = [[(e[0], e[1]) for e in f] for f in fls]

    def _link_slices(self):
        vw = self._voxel_width
        vd = self._voxel_depth
        fls = [[r] for r in self._slice_rois[0]]
        for i in self._progress(range(1, len(self._slice_rois)), desc="Linking slices",
                                position=self._progress_offset):
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
        self._fls = [[(z, r) for z, r in enumerate(f)] for f in fls]
        
    @property
    def path_lengths(self):
        return [np.linalg.norm(np.diff([list(np.array(r.centroid)*self._voxel_width) + [i*self._voxel_depth]
                                        for i, r in enumerate(f)], axis=0), axis=1).sum()
                for f in self._fls]
    
    def _fls_to_dict(self, zrs, do_shaft_outline=True, do_intensities=True):
        zs, rs = zip(*zrs)
        vw = self._voxel_width
        vd = self._voxel_depth
        straight = ((((np.array(rs[-1].centroid)-np.array(rs[0].centroid))*vw)**2).sum() +
                    ((zs[-1]-zs[0])*vd)**2)**0.5
        path = np.linalg.norm(np.diff(
            [[r.centroid[0]*vw, r.centroid[1]*vw, z*vd]
             for z, r in zrs],
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

        shaft_coordinates = [[roi.centroid[0]*vw, roi.centroid[1]*vw, z*vd]
                             for z, roi in zrs]
        fls_properties['shaft_coordinates'] = np.array(shaft_coordinates)
        fls_properties['X'] = shaft_coordinates[0][0]
        fls_properties['Y'] = shaft_coordinates[0][1]

        if do_shaft_outline:
            shaft_outline = []
            for i, roi in zrs:
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
            
            # shaft_mean_intensity = []
            # shaft_mean_background_intensity = []
            # shaft_std_background_intensity = []
        
            # for i, roi in zrs:
            #     base_indices = roi.coords
            #     foreground_mask = np.zeros_like(self._confocal[i], dtype=bool)
            #     x, y = base_indices[:, 0], base_indices[:, 1]
            #     foreground_mask[x, y] = True

            #     bounding_box = foreground_mask[x.min():x.max()+1, y.min():y.max()+1]
            #     dil1 = binary_dilation(foreground_mask, bounding_box)
            #     dil2 = binary_dilation(dil1, bounding_box)
            #     background_mask = dil2 & (~dil1) & (~self._roi_masks[i])
            #     nr_bg_pxls = background_mask.sum()

            #     shaft_mean_intensity.append(self._confocal[i][foreground_mask].mean())
            #     if nr_bg_pxls > 10:
            #         shaft_mean_background_intensity.append(self._confocal[i][background_mask].mean())
            #         shaft_std_background_intensity.append(self._confocal[i][background_mask].std())
            #     else:
            #         shaft_mean_background_intensity.append(float("NaN"))
            #         shaft_std_background_intensity.append(float("NaN"))
                
            # fls_properties['shaft_mean_intensity'] = np.array(shaft_mean_intensity)
            # fls_properties['shaft_mean_background_intensity'] = np.array(shaft_mean_background_intensity)
            # fls_properties['shaft_std_background_intensity'] = np.array(shaft_std_background_intensity)
        
        return fls_properties
    
    def get_table(self, *a, **kw):
        return pd.DataFrame([self._fls_to_dict(f, *a, **kw)
                             for f in self._progress(self._fls, desc='Table Output',
                                                     position=self._progress_offset)])

class FLS:
    fls_index = 0
    def __init__(self, frame, row):
        self.rows = []
        self.frames = []
        self.base_position = []
        self.add_row(frame, row)
        self.fls_index = FLS.fls_index
        FLS.fls_index += 1
    
    def add_row(self, frame, row):
        data = row.to_dict()
        data['frame'] = frame
        data['fls_index'] = self.fls_index
        self.rows.append(data)
        self.frames.append(frame)
        self.base_position.append(np.array([row.X, row.Y]))

class Frames:
    def __init__(self, images, base_slice,
                 voxel_width=0.1487,
                 voxel_depth=1.,
                 sigma=1.5,
                 K=3,
                 thresholding=threshold_triangle,
                 link_cutoff_distance=2.,
                 memory=4,
                 do_intensities=False,
                 do_shaft_outline=False,
                 alternative_z_linking=False,
                 alternative_time_linking=False,
                 progress=None, progress_offset=None):
        self.progress = progress
        self.progress_offset = progress_offset
        self.memory = 4

        def mapper(images):
            confocal = images['confocal']
            tirfs = {name: data for name, data in images.items() if name != 'confocal'}
            time = confocal.acquisition_time
            stack = Stack(confocal,
                          base_slice,
                          tirfs=tirfs,
                          voxel_width=voxel_width,
                          voxel_depth=voxel_depth,
                          sigma=sigma,
                          K=K,
                          thresholding=thresholding,
                          link_cutoff_distance=link_cutoff_distance,
                          alternative_linking=alternative_z_linking,
                          progress=progress,
                          progress_offset=progress_offset+1 if progress_offset is not None else None)
            return time, stack.get_table(do_intensities=do_intensities)
        self.stack_tables = [mapper(tp_imgs) for tp_imgs in progress(images, desc='Segmenting',
                                                                  position=progress_offset)]
        self.frame_times = [time for time, _ in self.stack_tables]
        
        if alternative_time_linking:
            self._alt_link_frames()
        else:
            self._link_frames()

    def _alt_link_frames(self):
        # Initial FLS collection is everything in the first frame
        flss = [FLS(0, row) for i, row in self.stack_tables[0][1].iterrows()]
        # Iterate over tables from time points, starting at the second one
        for i in self.progress(range(1, len(self.stack_tables)), desc='Framelink',
                               position=self.progress_offset):
            cands = self.stack_tables[i][1]
            nr_cand = cands.shape[0]
            if nr_cand == 0:
                continue

            rel_flss = [fls for fls in flss if fls.frames[-1] >= i-self.memory]
            nr_fls = len(rel_flss)
            if nr_fls == 0:
                for _, row in cands.iterrows():
                    flss.append(FLS(i, row))
                continue

            # Calculate cost matrix
            X = np.array([np.mean(fls.base_position, axis=0) for fls in rel_flss])
            Y = np.array([row.shaft_coordinates[0][:2] for _, row in cands.iterrows()])
            C = pairwise_distances(X, Y)
            xm, ym = np.unravel_index(np.argsort(C, axis=None), C.shape)
            candidates = C[xm, ym] < 1.
            xm = xm[candidates]
            ym = ym[candidates]

            looked_at_x = []
            looked_at_y = []
            for xi, yi in zip(xm, ym):
                if xi not in looked_at_x and yi not in looked_at_y:
                    looked_at_x.append(xi)
                    looked_at_y.append(yi)
                    rel_flss[xi].add_row(i, cands.iloc[yi])
            for yi in range(nr_cand):
                if yi not in looked_at_y:
                    flss.append(FLS(i, cands.iloc[yi]))
        self.flss = flss
        
    def _link_frames(self):
        def cost_matrix(flss, cands, frame, cost_thresh=1e3):
            # Relevant FLS are from the last 4 frames
            rel_flss = [fls for fls in flss if fls.frames[-1] >= frame-self.memory]
            if len(rel_flss) == 0:
                return [], cost_thresh+np.random.rand(cands.shape[0], cands.shape[0])*cost_thresh
            
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
        flss = [FLS(0, row) for i, row in self.stack_tables[0][1].iterrows()]
        # Iterate over tables from time points, starting at the second one
        for i in self.progress(range(1, len(self.stack_tables)), desc='Framelink',
                               position=self.progress_offset):
            tab = self.stack_tables[i][1]
            nr_cand = tab.shape[0]
            if nr_cand == 0:
                continue
            # Calculate cost matrix
            rel_flss, C = cost_matrix(flss, tab, i)
            # Find best assignments
            col_ind, row_ind, _ = lapjv.lapjv(C)
            nr_fls = len(rel_flss)
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
        proteins = []
        for f in self.flss:
            rows.extend(f.rows)
        return pd.DataFrame(rows)
            
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
