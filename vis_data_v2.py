# Visualize SynDatasetV2 samples (blur + noise + exposure steps)
from pathlib import Path
import shutil
import torch
import numpy as np
import cv2

from options.eld.train_options import TrainOptions
from dataset.lmdb_dataset import LMDBDataset
from dataset.sid_dataset import SynDatasetV2, worker_init_fn
import noise
from models.ELD_model import tensor2im


def main():
    root = Path(__file__).parent
    traindir = root / 'datasets' / 'train'
    output_root_dir = root / 'datasets' / 'vis_data_v2'
    shutil.rmtree(output_root_dir, ignore_errors=True)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    opt = TrainOptions().parse()
    noise_model = noise.NoiseModel(model=opt.noise, include=opt.include, exclude=None)

    repeat = 1 if opt.max_dataset_size is None else 1288 // opt.max_dataset_size

    ref_data = LMDBDataset(
        str(traindir / 'SID_Sony_Raw.db'),
        size=opt.max_dataset_size,
        repeat=repeat,
        return_meta=True,
    )

    dataset = SynDatasetV2(
        ref_dataset=ref_data,
        noise_model=noise_model,
        augment=False,
        seed=opt.seed,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=0, worker_init_fn=worker_init_fn)

    np.random.seed(0)
    for idx, batch in enumerate(dataloader):
        if idx >= 10:
            break

        y = batch['Y']
        x_t = batch['X_t_blur']
        x_ref = batch['X_ref']

        y_img = tensor2im(y, visualize=True)
        x_t_img = tensor2im(x_t, visualize=True)
        x_ref_img = tensor2im(x_ref, visualize=True)

        display = np.concatenate([
            y_img[:, :, ::-1],
            x_t_img[:, :, ::-1],
            x_ref_img[:, :, ::-1],
        ], axis=1).astype(np.uint8)

        outfile = output_root_dir / f'vis_v2_{idx:03d}.png'
        cv2.imwrite(str(outfile), display)


if __name__ == '__main__':
    main()
