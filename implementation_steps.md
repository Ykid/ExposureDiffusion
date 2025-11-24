Repo Scan + TODOs

Data (highest impact):

Build a blurred + dark RAW dataset following implementation_plan.md: use SID/ELD long-exposure RAW (datasets/SID/Sony/long) as X_ref, pack with dataset.sid_dataset.pack_raw_bayer; synthesize blur kernels (random motion 15–31 px, include identity) and convolve per Bayer channel to make X_blur.
Apply exposure drop (λ_T in [1/30,1/8]) and Poisson–Gaussian noise (match noise.NoiseModel gains/K from ISO) to form conditioned measurement Y; save metadata (λ_ref, λ_T, ISO, kernel id, K).
Add a new dataset class (or extend ELDTrainDataset) that returns (X_t_blur, Y, X_ref, λ_t, λ_T, λ_ref, ISO, kernel) for on-the-fly exposure-step sampling, mirroring SynDataset noise generation.
Prepare curated eval splits: reuse dataset/Sony_* lists plus a blurred set (see link_directory.sh) with fixed seeds/kernels; log kernels/ratios to make PSNR/SSIM comparable.
Dataset creation steps (offline prep)

Download SID RAW (Sony) → place under datasets/SID/Sony/{short,long} (matches current loaders).
Optional: create blurred copy tree like datasets/SID/Sony_blur_v2/{short,long} (see link_directory.sh for symlink pattern).
Generate LMDBs for speed (matching dataset/lmdb_dataset.py expectations):
Iterate long RAW files, pack to 4ch Bayer (pack_raw_bayer), store as uint16 with keys 00000000… and write meta_info.pkl containing {'shape': packed.shape, 'dtype': packed.dtype, idx: (wb, ccm, ISO, ratio)}; this becomes SID_Sony_Raw.db for targets.
For inputs, either (a) synthesize noise on-the-fly via SynDataset (simplest), or (b) pre-noise a copy to SID_Sony_syn_Raw_<camera>.db using noise.NoiseModel so training can skip per-batch sampling.
Keep pair lists (dataset/Sony_train.txt, Sony_val.txt, Sony_test*.txt) aligned with LMDB ordering; each line is short_fn long_fn.
Verify a few samples with vis_data.py (already set up to visualize LMDB/ELDTrainDataset) to ensure ratios/packing look correct.
If training in sRGB, also store wb, ccm, and optional CRF (see camera_params/ & process.read_wb_ccm) to allow ISP conversion.
Training loop changes

Modify Engine/model interface so set_input accepts both X_t_blur and conditioning Y + exposure scalars; propagate through loss so each step predicts X_ref (ED-style L1/Charbonnier with w_t).
Implement exposure schedule sampler {λ_t} inside the dataset (not per-iteration in the model) to avoid recomputation and ensure logging of λ_t per batch.
Add two-stage curriculum (no-blur then mixed blur) toggle in options; update options/eld/train_options.py defaults accordingly.
Update evaluation paths (test_full_SID.py, test_SID.py) to load new checkpoints and feed Y conditioning; keep crop logic for SID center 512×512.
Model/conditioning

Extend backbone (UNet/NAFNet in models/) to accept concatenated channels [X_t, Y] or FiLM conditioning; add embeddings for λ_t, λ_T, ISO, kernel id.
Keep Adaptive Residual Layer path; ensure ARL mask optionally conditioned on SNR from ratio/K.
Logging/monitoring

Save kernel/exposure/noise params with outputs in eval_results*.jsonl (see engine.append_jsonl) for ablations.
Add quick sanity tests: PSNR per step (iter_num sweep) and degradation-by-kernel plots.
If you want, I can draft the dataset builder (LMDB writer + blur/noise pipeline) and wire the new conditional dataloader before touching the training loop.