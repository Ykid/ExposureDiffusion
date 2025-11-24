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
from util import process


def packed_to_srgb_stub(tensor):
    """
    Cheap visualization: map packed Bayer 4ch to 3ch by averaging greens.
    """
    arr = tensor.detach()[0].cpu().float().numpy()
    arr = np.clip(arr, 0, 1)
    r = arr[0]
    g = 0.5 * (arr[1] + arr[3])
    b = arr[2]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255.0).astype(np.uint8)
    return rgb


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

    np.random.seed()
    for idx, batch in enumerate(dataloader):
        if idx >= 40:
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

        # sRGB-ish visualization
        wb = batch.get("wb", None)
        ccm = batch.get("ccm", None)

        def _to_numpy(meta_val):
            if meta_val is None:
                return None
            if isinstance(meta_val, torch.Tensor):
                return meta_val[0].cpu().numpy()
            return np.array(meta_val[0])

        wb_np = _to_numpy(wb)
        ccm_np = _to_numpy(ccm)

        def to_srgb(packed_tensor, gain=1.0):
            if wb_np is None or ccm_np is None:
                return packed_to_srgb_stub(packed_tensor * gain)
            arr = packed_tensor.detach()[0].cpu().float().numpy()
            arr = np.clip(arr * gain, 0, 1)
            srgb = process.raw2rgb_v2(arr, wb_np, ccm_np)
            return (srgb * 255.0).astype(np.uint8)

        lambda_t = float(batch.get("lambda_t", torch.tensor([1.0]))[0])
        lambda_T = float(batch.get("lambda_T", torch.tensor([1.0]))[0])
        lambda_ref = float(batch.get("lambda_ref", torch.tensor([1.0]))[0])

        gain_T = lambda_ref / max(lambda_T, 1e-6)
        gain_t = lambda_ref / max(lambda_t, 1e-6)

        y_srgb = to_srgb(y, gain=gain_T)
        x_t_srgb = to_srgb(x_t, gain=gain_t)
        x_ref_srgb = to_srgb(x_ref, gain=1.0)
        display_srgb = np.concatenate([y_srgb, x_t_srgb, x_ref_srgb], axis=1).astype(np.uint8)
        outfile_srgb = output_root_dir / f'vis_v2_srgb_{idx:03d}.png'
        cv2.imwrite(str(outfile_srgb), display_srgb)


if __name__ == '__main__':
    main()
