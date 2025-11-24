from posixpath import join
import torch.utils.data as data
import numpy as np
import pickle


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, noise_model=None, size=None, repeat=1, ratio_used_list=None, return_meta=False):
        import lmdb
        import multiprocessing
        import os
        self.db_path = db_path
        try:
            env_val = os.environ.get('ED_LMDB_CPU_COUNT')
            if env_val is not None:
                n = int(env_val)
                if n < 1:
                    raise ValueError("ED_LMDB_CPU_COUNT must be >= 1")
                self.num_cpus = n
            else:
                self.num_cpus = 1
        except Exception as e:
            print(f"[i] Warning: {e}")
            self.num_cpus = 1
        
        print(f'[i] LMDB num_cpus: {self.num_cpus}')
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.meta = pickle.load(open(join(db_path, 'meta_info.pkl'), 'rb'))
        self.shape = self.meta['shape']
        self.dtype = self.meta['dtype']
        self.return_meta = return_meta
        with self.env.begin(write=False) as txn:
            length = txn.stat()['entries']

        self.length = size or length
        if ratio_used_list is not None:
            idx_used = []
            for i in range(self.length):
                meta_item = self.meta[i]
                if isinstance(meta_item, dict):
                    ratio_val = meta_item.get("ratio", -1)
                else:
                    ratio_val = meta_item[3]
                if int(ratio_val) in ratio_used_list:
                    idx_used.append(i)
            print(f"Ratio used to train: {ratio_used_list}")
            print(f"Used pairs: {len(idx_used)} out of {self.length}")
        self.repeat = repeat
        self.noise_model = noise_model

    def __getitem__(self, index):
        env = self.env
        index = index % self.length
        
        with env.begin(write=False) as txn:
            raw_data = txn.get('{:08}'.format(index).encode('ascii'))

        flat_x = np.frombuffer(raw_data, self.dtype)
        x = flat_x.reshape(*self.shape)
        
        if self.dtype == np.uint16:
            x = np.clip(x / 65535, 0, 1).astype(np.float32)

        meta_entry = self.meta[index]
        meta_dict = {}
        if isinstance(meta_entry, dict):
            wb = meta_entry.get("wb", None)
            color_matrix = meta_entry.get("ccm", None)
            ISO = meta_entry.get("ISO", -1)
            ratio = meta_entry.get("ratio", meta_entry.get("lambda_ref", -1))
            meta_dict = dict(meta_entry)
        elif len(meta_entry) == 2:
            wb, color_matrix = meta_entry
            ratio, K, ISO = -1, -1, -1
            meta_dict = {"wb": wb, "ccm": color_matrix, "ISO": ISO, "ratio": ratio}
        else:
            wb, color_matrix, ISO, ratio = meta_entry
            if self.noise_model is not None:
                K = self.noise_model.ISO_to_K(ISO)
            else:
                K = -1
            meta_dict = {"wb": wb, "ccm": color_matrix, "ISO": ISO, "ratio": ratio}

        if self.noise_model is not None and isinstance(meta_entry, dict):
            K = self.noise_model.ISO_to_K(ISO)
        elif self.noise_model is None and "K" in meta_dict:
            K = meta_dict.get("K", -1)
        else:
            K = -1

        noise_info = {"ratio": ratio, "K": K, "ISO": ISO}
        if self.return_meta:
            merged_meta = {}
            merged_meta.update(meta_dict)
            merged_meta.update(noise_info)
            return x, merged_meta

        return x, noise_info # None: noise params

    def __len__(self):
        return int(self.length * self.repeat)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
