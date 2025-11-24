## **1. Data**

### **1.1 Base dataset**

Use a RAW low-light dataset with **paired long/short exposure**:

- e.g. SID (Sony subset) or ELD:
    - X_{\text{ref}}: long-exposure, sharp, well-exposed RAW (ground truth).
    - X_{\text{short}}: short-exposure noisy RAW (you can still use it for realism / noise statistics).

All processing is in **packed Bayer RAW** (4-channel).

### **1.2 Synthetic blur in RAW**

For each reference RAW X_{\text{ref}}:

1. **Sample a blur kernel** H:
    - Motion-blur PSF from random camera trajectories (e.g. 2D random walk projected to a small kernel, 15×15 or 31×31).
    - Optionally have a few “no blur” examples (identity kernel) so the model also sees pure low-light.
2. **Apply blur to the reference**:
    
    X_{\text{blur}} = H * X_{\text{ref}}
    
    Convolution done per Bayer channel.
    

### **1.3 Exposure drop + RAW noise (to create the condition** Y**)**

Choose a target low-light exposure \lambda_T and reference exposure \lambda_{\text{ref}} (from metadata or just set \lambda_{\text{ref}} = 1 and sample \lambda_T \in [1/30, 1/8]).

For each X_{\text{blur}}:

1. **Exposure drop**:
    
    X_{\text{low}} = \frac{\lambda_T}{\lambda_{\text{ref}}} X_{\text{blur}}
    
2. **Poisson–Gaussian RAW noise** (approximate ExposureDiffusion’s camera model):
    
    N = N_{\text{shot}} + N_{\text{read}}, \quad
    N_{\text{shot}} \sim \text{Poisson}(g X_{\text{low}}), \quad
    N_{\text{read}} \sim \mathcal{N}(0, \sigma_{\text{read}}^2)
    
    - g: gain depending on ISO.
    - \sigma_{\text{read}}: read noise std.
3. **Final observed RAW**:
    
    Y = X_{\text{low}} + N
    

So each training sample gives you:

- **Condition / measurement**: Y (blurred + dark + noisy RAW).
- **Target**: X_{\text{ref}} (clean, sharp RAW).
- **Known metadata**: \lambda_T / \lambda_{\text{ref}}, ISO, maybe kernel ID or blur strength.

---

## **2. Training strategy (ExposureDiffusion-style, but conditional)**

High-level: follow ExposureDiffusion’s **progressive exposure path**, but now the **denoiser is conditioned on** Y.

### **2.1 Exposure schedule and intermediate steps**

Define an exposure schedule \{\lambda_t\}_{t=0}^T:

- \lambda_0 = \lambda_{\text{ref}} (reference exposure).
- \lambda_T = \lambda_{\text{min}} (very low exposure).
- Decreasing sequence: \lambda_0 > \lambda_1 > \dots > \lambda_T.

For each step t, you have a **simulation of what the RAW would look like at exposure** \lambda_t **with blur**:

X_t^{(\text{blur})} = \frac{\lambda_t}{\lambda_{\text{ref}}} X_{\text{blur}} + N_t

with fresh Poisson–Gaussian noise N_t.

> This is exactly the ExposureDiffusion idea: you simulate a
> 
> 
> **physically meaningful exposure trajectory**
> 
> **blurred**
> 

### **2.2 Conditional denoiser**

Define a single network F_\theta (same backbone as ExposureDiffusion: NAFNet/UNet + Adaptive Residual Layer):

- Inputs:
    - noisy/low-exposure sample at step t: X_t^{(\text{blur})},
    - **condition**: Y (always at \lambda_T),
    - exposure info: \lambda_t, \lambda_T, \lambda_{\text{ref}}, ISO,
    - time step encoding t (or directly embed \lambda_t as “time”).
- Output:
    - predicted **reference RAW** \hat{X}^{\text{ref}}_t,
    - plus residual branch / ARL mask exactly like ExposureDiffusion.

Conditioning trick:

- Easiest: concatenate [X_t^{(\text{blur})}, Y] along the channel dimension.
- Better: encode Y with a shallow encoder and inject via FiLM / cross-attention, but you don’t have to specify that now.

### **2.3 Per-iteration training step**

For each SGD iteration:

1. Sample a training example:
    - X_{\text{ref}} from dataset.
    - Generate X_{\text{blur}}, Y as above.
2. Sample a random step t \in \{1,\dots,T\}.
3. Generate **blurred low-exposure sample** X_t^{(\text{blur})} at exposure \lambda_t.
4. Forward pass:
    
    \hat{X}^{\text{ref}}_t = F_\theta\big(X_t^{(\text{blur})}, Y, \lambda_t, \lambda_T, \lambda_{\text{ref}}, \text{ISO}, t\big)
    
5. Compute ExposureDiffusion-style loss (see next section).
6. Backprop and update \theta.

Optional schedule trick (recommended):

- **Two-stage training**:
    1. Stage 1: train on pure low-light (no blur: use H = I) to match original ExposureDiffusion.
    2. Stage 2: fine-tune with random blur kernels H (mix blurred and unblurred) so the prior doesn’t forget the base low-light task.

---

## **3. Training loss (ExposureDiffusion-style)**

We keep the **same flavour** as ExposureDiffusion: **predict reference RAW at each exposure step** and penalize deviation with an L1 (or Charbonnier) loss, possibly weighted over t.

For a mini-batch, loss is:

L = \mathbb{E}_{X_{\text{ref}}, Y, t}
\left[
w_t \left\| \hat{X}^{\text{ref}}_t - X_{\text{ref}} \right\|_1
\right]

where:

- \hat{X}^{\text{ref}}_t = F_\theta(X_t^{(\text{blur})}, Y, \dots),
- w_t are step weights (e.g. uniform or following ED’s KL-based weights).

If you follow the original paper more closely:

- They derive an **upper bound on KL** between the model and true exposure process, which decomposes into **step-wise reconstruction terms**; your conditional variant is the same except that the denoiser sees Y.

You still keep:

- **Adaptive Residual Layer (ARL)** internally:
    - F_\theta learns to blend between direct prediction and residual denoising based on local SNR.
- Optional auxiliary losses (like small ISP + VGG on sRGB previews) can be added, but the **core is per-step L1 to** X_{\text{ref}}.

No explicit blur data-consistency term is needed here, since blur is already baked into the simulated X_t^{(\text{blur})} and Y; you’re sticking to the **pure “ED-style” clean-space reconstruction loss**.

---

## **4. Inference step (joint deblurring + low-light with one model)**

At test time you have only:

- observed blurred + dark RAW Y_{\text{test}},
- plus its metadata (exposure \lambda_T, ISO, maybe approximate H if you want to simulate more steps, but the model has already been trained to handle blur baked into Y).

You want to output \hat{X}_{\text{ref}} (sharp, well-exposed RAW).

### **4.1 Initialization**

Define the same exposure schedule \{\lambda_t\}_{t=0}^T as in training.

- Initialize X_T as a simple **amplified version** of Y_{\text{test}}:
    
    X_T = \frac{\lambda_{\text{ref}}}{\lambda_T} Y_{\text{test}}
    
    (clipped to valid RAW range).
    

This is a crude “physics-based” guess at what the reference exposure might look like.

### **4.2 Progressive ExposureDiffusion-style sampling**

For t = T, T-1, \dots, 1:

1. **Denoising / exposure step (conditional)**
    
    Use your model to predict reference RAW from the current low-exposure sample:
    
    \hat{X}^{\text{ref}}_t = F_\theta\big(X_t, Y_{\text{test}}, \lambda_t, \lambda_T, \lambda_{\text{ref}}, \text{ISO}, t\big)
    
2. **Update rule (ED-style)**
    
    ExposureDiffusion uses a deterministic or mildly stochastic update of the form:
    
    X_{t-1} = \text{Update}(X_t, \hat{X}^{\text{ref}}_t, \lambda_t, \lambda_{t-1})
    
    In your conditional version you can use the same structure as ED:
    
    - treat \hat{X}^{\text{ref}}_t as the estimate of the clean image,
    - combine it with X_t according to their derived “bridging” formula (or simply move towards \hat{X}^{\text{ref}}_t with a small residual + optional noise).
    
    If you want a simple deterministic sampler to start with:
    
    - ignore added noise and set, for example:
        
        X_{t-1} = X_t + \alpha_t \big(\hat{X}^{\text{ref}}_t - X_t\big)
        
        with \alpha_t increasing as you go to cleaner exposures.
        
3. After the loop, take:
    
    \hat{X}_{\text{final}} = \hat{X}^{\text{ref}}_1
    
    or X_0 depending on how you define the last step.
    

### **4.3 ISP to visualize**

- Demosaic, white-balance, tone-map \hat{X}_{\text{final}} once at the end to get sRGB.
- All blur + low-light restoration lives in RAW-space.

---

### **TL;DR**

- **Data**: SID/ELD RAW pairs; for each X_{\text{ref}}, synthesize blur H in RAW and then low-light + Poisson–Gaussian noise to get Y.
- **Training strategy**: define an exposure schedule \lambda_t, simulate blurred low-exposure samples X_t^{(\text{blur})} along this path, and train a **single ED-style denoiser** F_\theta that is **conditioned on** Y.
- **Loss**: ExposureDiffusion-style per-step L1 (or Charbonnier) between \hat{X}^{\text{ref}}_t and X_{\text{ref}}, optionally with KL-based weights w_t.
- **Inference**: initialize from amplified Y, run a progressive ED-style sampling loop where each step uses F_\theta(\cdot, Y) to move towards a clean, well-exposed RAW; then run ISP once to visualize.

This gives you **one conditional diffusion model** that jointly handles deblurring and low-light enhancement, while staying faithful to the ExposureDiffusion loss philosophy.