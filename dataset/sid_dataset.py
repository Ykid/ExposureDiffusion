# See in the Dark (SID) dataset
import torch
import os
import rawpy
import numpy as np
from os.path import join
import dataset.torchdata as torchdata
import util.process as process
from util.util import loadmat
import exifread
import pickle
import random
from torchvision.utils import save_image 

BaseDataset = torchdata.Dataset
from dataset.blur_utils import random_motion_kernel, apply_kernel_bayer, add_poisson_gaussian_noise, sample_lambda_pair, lambda_range_from_exposure


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


class SIDDataset(BaseDataset):
    def __init__(
        self, datadir, paired_fns, noise_maker, size=None, flag=None, augment=True, repeat=1, cfa='bayer', memorize=True, 
        stage_in='raw', stage_out='raw', gt_wb=False, CRF=None, patch_size=512):
        super(SIDDataset, self).__init__()
        assert cfa == 'bayer' or cfa == 'xtrans'
        self.size = size
        self.noise_maker = noise_maker
        self.datadir = datadir
        self.paired_fns = paired_fns
        self.flag = flag
        self.augment = augment
        self.patch_size = patch_size
        self.repeat = repeat
        self.cfa = cfa

        self.pack_raw = pack_raw_bayer if cfa == 'bayer' else pack_raw_xtrans

        assert stage_in in ['raw', 'srgb']
        assert stage_out in ['raw', 'srgb']                
        self.stage_in = stage_in
        self.stage_out = stage_out
        self.gt_wb = gt_wb     
        self.CRF = CRF   

        if size is not None:
            self.paired_fns = self.paired_fns[:size]
        
        self.memorize = memorize
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, i):
        i = i % len(self.paired_fns)
        input_fn, target_fn = self.paired_fns[i]

        input_path = join(self.datadir, 'short', input_fn)
        target_path = join(self.datadir, 'long', target_fn)
        
        iso, expo = metainfo(input_path)
        K = self.noise_maker.ISO_to_K(iso)
        
        ratio = compute_expo_ratio(input_fn, target_fn)       
        CRF = self.CRF         

        if self.memorize:
            if target_fn not in self.target_dict:
                with rawpy.imread(target_path) as raw_target:                    
                    target_image = self.pack_raw(raw_target)    
                    wb, ccm = process.read_wb_ccm(raw_target)
                    if self.stage_out == 'srgb':
                        target_image = process.raw2rgb(target_image, raw_target, CRF)
                    self.target_dict[target_fn] = target_image
                    self.target_dict_aux[target_fn] = (wb, ccm)

            if input_fn not in self.input_dict:
                with rawpy.imread(input_path) as raw_input:
                    input_image = self.pack_raw(raw_input) * ratio
                    if self.stage_in == 'srgb':
                        if self.gt_wb:
                            wb, ccm = self.target_dict_aux[target_fn]
                            input_image = process.raw2rgb_v2(input_image, wb, ccm, CRF)
                        else:
                            input_image = process.raw2rgb(input_image, raw_input, CRF)
                    self.input_dict[input_fn] = input_image

            input_image = self.input_dict[input_fn]
            target_image = self.target_dict[target_fn]
            (wb, ccm) = self.target_dict_aux[target_fn]
        else:
            with rawpy.imread(target_path) as raw_target:                    
                target_image = self.pack_raw(raw_target)    
                wb, ccm = process.read_wb_ccm(raw_target)
                if self.stage_out == 'srgb':
                    target_image = process.raw2rgb(target_image, raw_target, CRF)

            with rawpy.imread(input_path) as raw_input:
                input_image = self.pack_raw(raw_input) * ratio
                if self.stage_in == 'srgb':
                    if self.gt_wb:
                        input_image = process.raw2rgb_v2(input_image, wb, ccm, CRF)
                    else:
                        input_image = process.raw2rgb(input_image, raw_input, CRF)  

        if self.augment:
            H = input_image.shape[1]
            W = target_image.shape[2]

            ps = self.patch_size

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            input = input_image[:, yy:yy + ps, xx:xx + ps]
            target = target_image[:, yy:yy + ps, xx:xx + ps]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input = np.flip(input, axis=1) # H
                target = np.flip(target, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input = np.flip(input, axis=2) # W
                target = np.flip(target, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input = np.transpose(input, (0, 2, 1))
                target = np.transpose(target, (0, 2, 1))
        else:
            input = input_image
            target = target_image

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        # if True:
        #     assert K > 0 and ratio > 0
        #     saturation_level = 16383 - 800
        #     target_ratio = ratio // 20
            
        #     input_photon = (input * saturation_level / ratio / K)
        #     increased_photon = np.random.poisson(target * saturation_level / ratio / K * (target_ratio-1))
        #     input = (increased_photon + input_photon) * K /saturation_level / target_ratio * ratio
        #     input = input.astype("float32")
        #     ratio = ratio // target_ratio
            
        dic =  {'input': input, 'target': target, 'fn': input_fn, 'cfa': self.cfa, 'rawpath': target_path, "ratio": ratio, "K":K}

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        return len(self.paired_fns) * self.repeat


def compute_expo_ratio(input_fn, target_fn):        
    in_exposure = float(input_fn.split('_')[-1].split("s.")[0])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio


def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = 16383
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out


def pack_raw_xtrans(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = (im - 1024) / (16383 - 1024)  # subtract the black level
    im = np.clip(im, 0, 1)

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((9, H // 3, W // 3), dtype=np.float32)

    # 0 R
    out[0, 0::2, 0::2] = im[0:H:6, 0:W:6]
    out[0, 0::2, 1::2] = im[0:H:6, 4:W:6]
    out[0, 1::2, 0::2] = im[3:H:6, 1:W:6]
    out[0, 1::2, 1::2] = im[3:H:6, 3:W:6]

    # 1 G
    out[1, 0::2, 0::2] = im[0:H:6, 2:W:6]
    out[1, 0::2, 1::2] = im[0:H:6, 5:W:6]
    out[1, 1::2, 0::2] = im[3:H:6, 2:W:6]
    out[1, 1::2, 1::2] = im[3:H:6, 5:W:6]

    # 1 B
    out[2, 0::2, 0::2] = im[0:H:6, 1:W:6]
    out[2, 0::2, 1::2] = im[0:H:6, 3:W:6]
    out[2, 1::2, 0::2] = im[3:H:6, 0:W:6]
    out[2, 1::2, 1::2] = im[3:H:6, 4:W:6]

    # 4 R
    out[3, 0::2, 0::2] = im[1:H:6, 2:W:6]
    out[3, 0::2, 1::2] = im[2:H:6, 5:W:6]
    out[3, 1::2, 0::2] = im[5:H:6, 2:W:6]
    out[3, 1::2, 1::2] = im[4:H:6, 5:W:6]

    # 5 B
    out[4, 0::2, 0::2] = im[2:H:6, 2:W:6]
    out[4, 0::2, 1::2] = im[1:H:6, 5:W:6]
    out[4, 1::2, 0::2] = im[4:H:6, 2:W:6]
    out[4, 1::2, 1::2] = im[5:H:6, 5:W:6]

    out[5, :, :] = im[1:H:3, 0:W:3]
    out[6, :, :] = im[1:H:3, 1:W:3]
    out[7, :, :] = im[2:H:3, 0:W:3]
    out[8, :, :] = im[2:H:3, 1:W:3]
    return out


class SynDataset(BaseDataset):  # generate noisy image only 
    def __init__(self, dataset, size=None, flag=None, noise_maker=None, repeat=1, cfa='bayer', num_burst=1, continuous_noise=False):
        super(SynDataset, self).__init__()        
        self.size = size
        self.dataset = dataset
        self.flag = flag
        self.repeat = repeat
        self.noise_maker = noise_maker
        self.cfa = cfa
        self.num_burst = num_burst
        self.continuous_noise = continuous_noise
        
    def __getitem__(self, i):
        if self.size is not None:
            i = i % self.size
        else:
            i = i % len(self.dataset)
            
        data, metadata = self.dataset[i]

        if self.num_burst > 1:            
            inputs = []
            params = self.noise_maker._sample_params()     
            for k in range(self.num_burst):           
                # inputs.append(self.noise_maker(data))
                inputs.append(self.noise_maker(data, params=params, continuous=self.continuous_noise))
            input = np.concatenate(inputs, axis=0)
        else:
            input, params = self.noise_maker(data, continuous=self.continuous_noise)
        
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        return input, params
        
    def __len__(self):
        size = self.size or len(self.dataset)
        return int(size * self.repeat)
    

class ISPDataset(BaseDataset):
    def __init__(self, dataset, noise_maker=None, cfa='bayer', meta_info=None, CRF=None):
        super(ISPDataset, self).__init__()        
        self.dataset = dataset
        self.noise_maker = noise_maker
        self.cfa = cfa

        if meta_info is None:
            self.meta_info = dataset.meta
        else:
            self.meta_info = meta_info

        self.CRF = CRF
        
    def __getitem__(self, i):
        data = self.dataset[i]
        (wb, ccm) = self.meta_info[i]
        CRF = self.CRF
        
        if self.noise_maker is not None:        
            input = self.noise_maker(data)
        else:
            input = data

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = process.raw2rgb_v2(input, wb, ccm, CRF)
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        return input

    def __len__(self):
        return len(self.dataset)    


class ELDTrainDataset(BaseDataset):
    def __init__(self, target_dataset, input_datasets, size=None, flag=None, augment=True, cfa='bayer', syn_noise=False):
        super(ELDTrainDataset, self).__init__()
        self.size = size
        self.target_dataset = target_dataset
        self.input_datasets = input_datasets
        self.flag = flag
        self.augment = augment
        self.cfa = cfa
        self.syn_noise = syn_noise # synthetic possion noise

    def __getitem__(self, i):
        N = len(self.input_datasets)
        input_image, noise_params = self.input_datasets[i%N][i//N]
        target_image, _ = self.target_dataset[i//N]        

        target = target_image 
        input = input_image       
    
        if self.augment:
            W = target_image.shape[2]
            H = target_image.shape[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
                input = np.flip(input, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
                input = np.flip(input, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))
                input = np.transpose(input, (0, 2, 1))        

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)
        
        ratio = noise_params["ratio"]
        K = noise_params["K"]

        if self.syn_noise:
            assert K > 0 and ratio > 0
            saturation_level = 16383 - 800
            target_ratio = 1 / random.uniform(1/ratio, 1/100)
            
            input_photon = (input * saturation_level / ratio / K)
            increased_photon = np.random.poisson(target * saturation_level / ratio / K * (ratio/target_ratio-1))
            input = (increased_photon + input_photon) * K /saturation_level * target_ratio
            input = input.astype("float32")
        
        dic =  {'input': input, 'target': target, "ratio": ratio, "K": K} 
        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        size = self.size or len(self.target_dataset) * len(self.input_datasets)
        return size


# class ConditionalExposureDataset(BaseDataset):
#     """
#     Dataset that returns blurred exposure steps and a conditioned measurement Y.
#     Outputs:
#         X_t_blur: blurred + noisy sample at lambda_t
#         Y: blurred + noisy measurement at lambda_T
#         X_ref: clean reference patch
#         lambda_t, lambda_T, lambda_ref, ISO, kernel, kernel_id
#     """
#     def __init__(
#         self,
#         ref_dataset,
#         noise_model,
#         lambda_T_range=(1/30, 1/8),
#         lambda_ref=1.0,
#         kernel_size_range=(15, 31),
#         identity_prob=0.05,
#         sample_log_exposure=True,
#         augment=False,
#         seed=None,
#     ):
#         super(ConditionalExposureDataset, self).__init__()
#         self.ref_dataset = ref_dataset
#         self.noise_model = noise_model
#         self.lambda_T_range = lambda_T_range
#         self.lambda_ref = lambda_ref
#         self.kernel_size_range = kernel_size_range
#         self.identity_prob = identity_prob
#         self.sample_log_exposure = sample_log_exposure
#         self.augment = augment
#         self.rng = np.random.default_rng(seed)

#     def _sample_noise_params(self, iso):
#         if self.noise_model is None:
#             return {
#                 "K": 1.0,
#                 "g_scale": 0.0,
#                 "sigma_r": 0.0,
#                 "saturation_level": 16383 - 800,
#             }
#         return self.noise_model.sample_params_for_iso(
#             ISO=iso,
#             ratio=self.lambda_ref,
#             continuous=True,
#         )

#     def _maybe_augment(self, input_arrs):
#         if not self.augment:
#             return input_arrs
#         if np.random.randint(2, size=1)[0] == 1:
#             input_arrs = [np.flip(x, axis=1) for x in input_arrs]
#         if np.random.randint(2, size=1)[0] == 1:
#             input_arrs = [np.flip(x, axis=2) for x in input_arrs]
#         if np.random.randint(2, size=1)[0] == 1:
#             input_arrs = [np.transpose(x, (0, 2, 1)) for x in input_arrs]
#         return input_arrs

#     def __getitem__(self, i):
#         x_ref, meta = self.ref_dataset[i]
#         print("meta:", meta)
#         iso = -1
#         wb = None
#         ccm = None
#         exposure = None
#         lambda_range_meta = None
#         if isinstance(meta, dict):
#             iso = meta.get("ISO", meta.get("iso", -1))
#             wb = meta.get("wb", None)
#             ccm = meta.get("ccm", None)
#             exposure = meta.get("exposure", None)
#             lambda_range_meta = meta.get("lambda_T_range", None)
#         elif isinstance(meta, (list, tuple)) and len(meta) >= 3:
#             iso = meta[2]
#             wb = meta[0]
#             ccm = meta[1]
#         iso = 100 if iso is None or iso < 0 else iso

#         kernel, kernel_id = random_motion_kernel(
#             kernel_size_range=self.kernel_size_range,
#             identity_prob=self.identity_prob,
#             rng=self.rng,
#         )
#         x_blur = apply_kernel_bayer(x_ref, kernel)

#         lambda_t, lambda_T = sample_lambda_pair(
#             lambda_ref=self.lambda_ref,
#             lambda_T_range=self.lambda_T_range,
#             log_space=self.sample_log_exposure,
#             rng=self.rng,
#         )

#         params_T = self._sample_noise_params(iso)
#         params_t = self._sample_noise_params(iso)
#         y = add_poisson_gaussian_noise(
#             x_blur * (lambda_T / self.lambda_ref),
#             params_T,
#             rng=self.rng,
#         )
#         x_t_blur = add_poisson_gaussian_noise(
#             x_blur * (lambda_t / self.lambda_ref),
#             params_t,
#             rng=self.rng,
#         )

#         x_t_blur, y, x_ref_clean, x_blur_clean = [
#             np.ascontiguousarray(arr.astype(np.float32))
#             for arr in self._maybe_augment([x_t_blur, y, x_ref, x_blur])
#         ]

#         return {
#             "X_t_blur": x_t_blur,
#             "Y": y,
#             "X_ref": x_ref_clean,
#             "X_blur": x_blur_clean,
#             "lambda_t": float(lambda_t),
#             "lambda_T": float(lambda_T),
#             "lambda_ref": float(self.lambda_ref),
#             "ISO": iso,
#             "kernel": kernel,
#             "kernel_id": kernel_id,
#             "noise_params_t": params_t,
#             "noise_params_T": params_T,
#             "wb": wb,
#             "ccm": ccm,
#         }

#     def __len__(self):
#         return len(self.ref_dataset)


class SynDatasetV2(BaseDataset):
    """
    Blur + noise synthesizer that wraps a clean reference dataset (e.g., LMDBDataset).
    Returns the full tuple needed by the conditional training loop.
    """
    def __init__(
        self,
        ref_dataset,
        noise_model,
        lambda_T_range=(1/30, 1/8),
        lambda_ref=1.0,
        kernel_size_range=(15, 31),
        identity_prob=0.05,
        sample_log_exposure=True,
        augment=False,
        seed=None,
    ):
        super(SynDatasetV2, self).__init__()
        self.ref_dataset = ref_dataset
        self.noise_model = noise_model
        self.lambda_T_range = lambda_T_range
        self.lambda_ref = lambda_ref
        self.kernel_size_range = kernel_size_range
        self.identity_prob = identity_prob
        self.sample_log_exposure = sample_log_exposure
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def _sample_noise_params(self, iso, ratio=None):
        if self.noise_model is None:
            return {
                "K": 1.0,
                "g_scale": 0.0,
                "sigma_r": 0.0,
                "saturation_level": 16383 - 800,
                "ratio": ratio if ratio is not None else self.lambda_ref,
            }
        print("Sampling noise params for ISO:", iso, "ratio:", ratio if ratio is not None else self.lambda_ref)
        return self.noise_model.sample_params_for_iso(
            ISO=iso,
            ratio=ratio if ratio is not None else self.lambda_ref,
            continuous=True,
        )

    def _maybe_augment(self, arrs):
        if not self.augment:
            return arrs
        if np.random.randint(2, size=1)[0] == 1:
            arrs = [np.flip(x, axis=1) for x in arrs]
        if np.random.randint(2, size=1)[0] == 1:
            arrs = [np.flip(x, axis=2) for x in arrs]
        if np.random.randint(2, size=1)[0] == 1:
            arrs = [np.transpose(x, (0, 2, 1)) for x in arrs]
        return arrs

    def __getitem__(self, i):
        x_ref, meta = self.ref_dataset[i]
        iso = -1
        wb = None
        ccm = None
        exposure = None
        lambda_range_meta = None
        lambda_ref_local = self.lambda_ref
        if isinstance(meta, dict):
            iso = meta.get("ISO", meta.get("iso", -1))
            wb = meta.get("wb", None)
            ccm = meta.get("ccm", None)
            exposure = meta.get("exposure", None)
            lambda_range_meta = meta.get("lambda_T_range", None)
            lambda_ref_local = meta.get("lambda_ref", lambda_ref_local)
        elif isinstance(meta, (list, tuple)) and len(meta) >= 3:
            iso = meta[2]
            wb = meta[0]
            ccm = meta[1]
        # print("meta:", meta)
        # print("exposure:", exposure)
        iso = 100 if iso is None or iso < 0 else iso


        if exposure is not None:
            lambda_range = lambda_range_from_exposure(exposure)
        else:
            # print("using default lambda range")
            lambda_range = self.lambda_T_range

        kernel, kernel_id = random_motion_kernel(
            kernel_size_range=self.kernel_size_range,
            identity_prob=self.identity_prob,
            rng=self.rng,
        )
        x_blur = apply_kernel_bayer(x_ref, kernel)

        # print("lambda range:", lambda_range, "exposure:", exposure, "lambda_ref_local:", lambda_ref_local)
        lambda_t, lambda_T = sample_lambda_pair(
            lambda_ref=lambda_ref_local,
            lambda_T_range=lambda_range,
            log_space=self.sample_log_exposure,
            rng=self.rng,
        )

        # Condition measurement at lambda_T, step sample at lambda_t
        params_T = self._sample_noise_params(iso, ratio=lambda_ref_local / lambda_T)
        params_t = self._sample_noise_params(iso, ratio=lambda_ref_local / lambda_t)
        print("lambda_T_range:", lambda_T, "lambda_t:", lambda_t, 'lambda_ref_local:', lambda_ref_local)
        y = add_poisson_gaussian_noise(
            x_blur * (lambda_T / lambda_ref_local),
            params_T,
            rng=self.rng,
        )
        x_t_blur = add_poisson_gaussian_noise(
            x_blur * (lambda_t / lambda_ref_local),
            params_t,
            rng=self.rng,
        )

        x_t_blur, y, x_ref_clean, x_blur_clean = [
            np.ascontiguousarray(arr.astype(np.float32))
            for arr in self._maybe_augment([x_t_blur, y, x_ref, x_blur])
        ]

        return {
            "X_t_blur": x_t_blur,
            "Y": y,
            "X_ref": x_ref_clean,
            "X_blur": x_blur_clean,
            "lambda_t": float(lambda_t),
            "lambda_T": float(lambda_T),
            "lambda_ref": float(lambda_ref_local),
            "ISO": iso,
            "kernel": kernel,
            "kernel_id": kernel_id,
            "noise_params_t": params_t,
            "noise_params_T": params_T,
            "wb": wb,
            "ccm": ccm,
            "exposure" : exposure,
        }

    def __len__(self):
        return len(self.ref_dataset)


class ELDEvalDataset(BaseDataset):
    def __init__(self, basedir, camera_suffix, noiser_maker, scenes=None, img_ids=None):
        super(ELDEvalDataset, self).__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix # ('Canon', '.CR2')
        self.scenes = scenes
        self.img_ids = img_ids
        # self.input_dict = {}
        # self.target_dict = {}
        self.noise_maker = noiser_maker
        
    def __getitem__(self, i):
        camera, suffix = self.camera_suffix
        
        scene_id = i // len(self.img_ids)
        img_id = i % len(self.img_ids)

        scene = 'scene-{}'.format(self.scenes[scene_id])

        datadir = join(self.basedir, camera, scene)

        input_path = join(datadir, 'IMG_{:04d}{}'.format(self.img_ids[img_id], suffix))

        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        
        target_path = join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

        iso, expo = metainfo(target_path)
        target_expo = iso * expo
        iso, expo = metainfo(input_path)

        ratio = target_expo / (iso * expo)
        K = self.noise_maker.ISO_to_K(iso)
        with rawpy.imread(input_path) as raw:
            input = pack_raw_bayer(raw) * ratio            

        with rawpy.imread(target_path) as raw:
            target = pack_raw_bayer(raw)

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)        

        data = {'input': input, 'target': target, 'fn':input_path, 'rawpath': target_path, 'ratio': ratio, "K": K}
        
        return data

    def __len__(self):
        return len(self.scenes) * len(self.img_ids)
