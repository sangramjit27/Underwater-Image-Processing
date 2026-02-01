# run_pipeline_superclarity_autotune.py
"""
Folder-based underwater enhancement:
 - Uses MSRCR + gray-world + CLAHE + bilateral + unsharp + saturation.
 - Auto-tunes a small grid of parameters per-image and picks best by a perceptual score
 - Works on folders: data/{train,val,test}/input (and compares to output if present)
 - Saves results to results/enhanced_images/{train,val,test}/
 - Writes metrics CSV if GT present.
"""

import os
import glob
import shutil
import math
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd
import argparse
from collections import namedtuple

# ---- CONFIG ----
IMG_SIZE = 512  # resize processing
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENH_DIR = os.path.join(RESULTS_DIR, "enhanced_images")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_report.csv")
FOLDER_LIST = [("train","input","output"), ("val","input","output"), ("test","input","output")]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENH_DIR, exist_ok=True)
for name,a,b in FOLDER_LIST:
    os.makedirs(os.path.join(DATA_DIR, name, a), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, name, b), exist_ok=True)
    os.makedirs(os.path.join(ENH_DIR, name), exist_ok=True)

# ---- Helpers ----
def load_rgb(path):
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil)  # RGB uint8
    return arr

def save_rgb(path, arr_rgb_uint8):
    # arr in RGB uint8
    bgr = cv2.cvtColor(arr_rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def resize_rgb(arr, size=(IMG_SIZE, IMG_SIZE)):
    return cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)

# ---- Metrics used for auto-tune selection ----
def colorfulness_metric(img_rgb_uint8):
    # Hasler & Süsstrunk metric
    img = img_rgb_uint8.astype(np.float32)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    rg = R - G
    yb = 0.5*(R + G) - B
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    std_root = math.sqrt(std_rg**2 + std_yb**2)
    mean_root = math.sqrt(mean_rg**2 + mean_yb**2)
    return std_root + 0.3 * mean_root

def entropy_metric(img_rgb_uint8):
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    ent = -np.sum([p * math.log2(p) for p in hist if p>0])
    return ent

# final combined score to maximise when GT not available
def combined_natural_score(img_rgb_uint8):
    # weight colorfulness and entropy (encourages color & detail)
    cf = colorfulness_metric(img_rgb_uint8)
    ent = entropy_metric(img_rgb_uint8)
    # normalize roughly (empirical)
    return 0.6 * (cf) + 0.4 * (ent)

# ---- Image processing building blocks ----
def gray_world_balance(img_rgb_uint8):
    img = img_rgb_uint8.astype(np.float32)
    mean = img.mean(axis=(0,1))
    mean_gray = mean.mean()
    scale = mean_gray / (mean + 1e-8)
    out = np.clip(img * scale.reshape(1,1,3), 0, 255).astype(np.uint8)
    return out

def apply_clahe_on_l(img_rgb_uint8, clipLimit=3.0, tileGridSize=(8,8)):
    # Convert to LAB and apply CLAHE to L channel
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return rgb2

def unsharp_mask(img_uint8, amount=1.0, radius=1.2):
    sigma = radius
    blurred = cv2.GaussianBlur(img_uint8, (0,0), sigma)
    sharp = cv2.addWeighted(img_uint8, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def bilateral_filter_rgb(img_uint8, d=9, sigmaColor=50, sigmaSpace=50):
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    bgr2 = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return rgb2

# ---- MSRCR (Multi-Scale Retinex with Color Restoration) implementation ----
def single_scale_retinex(channel, sigma):
    # channel float (0..1)
    blur = cv2.GaussianBlur((channel*255.0).astype(np.uint8), (0,0), sigma).astype(np.float32)/255.0
    # avoid log domain issues
    ret = np.log1p(channel) - np.log1p(blur)
    return ret

def multi_scale_retinex(img_rgb_uint8, sigmas=(15,80,250), weights=None):
    # input RGB uint8 -> returns float image scaled to [0,1] after dynamic range compression
    if weights is None:
        weights = [1.0/len(sigmas)]*len(sigmas)
    img_f = img_rgb_uint8.astype(np.float32)/255.0
    retinex = np.zeros_like(img_f)
    for c in range(3):
        channel = img_f[:,:,c]
        msr = np.zeros_like(channel)
        for s,w in zip(sigmas,weights):
            msr += w * single_scale_retinex(channel, s)
        retinex[:,:,c] = msr
    # color restoration
    # compute color gain
    sum_rgb = np.sum(img_f, axis=2) + 1e-8
    for c in range(3):
        retinex[:,:,c] = retinex[:,:,c] * (np.log1p(125 * (img_f[:,:,c] / sum_rgb)))
    # scale to 0..1 via simple normalization
    v_min = retinex.min()
    v_max = retinex.max()
    out = (retinex - v_min) / (v_max - v_min + 1e-12)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

# ---- Full pipeline per-image: tries parameter variations and selects best ----
ParamSet = namedtuple("ParamSet", ["msr_sigmas", "clahe_clip", "bilateral_d", "unsharp_amount", "saturation_mult", "lap_strength"])

def enhance_candidate(img_rgb_uint8, params: ParamSet):
    # 1. MSRCR
    msr = multi_scale_retinex(img_rgb_uint8, sigmas=params.msr_sigmas)
    # 2. Gray-world color balance
    gw = gray_world_balance(msr)
    # 3. CLAHE on L
    clahe = apply_clahe_on_l(gw, clipLimit=params.clahe_clip, tileGridSize=(8,8))
    # 4. Laplacian-like boost (local contrast)
    img_f = clahe.astype(np.float32)/255.0
    blurred = cv2.GaussianBlur((img_f*255).astype(np.uint8),(5,5),0).astype(np.float32)/255.0
    lap = img_f - blurred
    boosted = np.clip(img_f + params.lap_strength * lap, 0.0, 1.0)
    boosted_uint8 = (boosted*255.0).astype(np.uint8)
    # 5. Bilateral smooth to reduce noise but keep edges
    smooth = bilateral_filter_rgb(boosted_uint8, d=params.bilateral_d, sigmaColor=75, sigmaSpace=75)
    # 6. Unsharp mask to bring back crispness
    sharp = unsharp_mask(smooth, amount=params.unsharp_amount, radius=1.2)
    # 7. Saturation boost in HSV
    hsv = cv2.cvtColor(sharp, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * params.saturation_mult, 0, 255)
    out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out_rgb

def auto_tune_and_enhance(img_rgb_uint8):
    # grid of parameter candidates - keep small to be fast
    candidate_params = [
        ParamSet(msr_sigmas=(15,80,250), clahe_clip=2.0, bilateral_d=7, unsharp_amount=0.8, saturation_mult=1.1, lap_strength=0.9),
        ParamSet(msr_sigmas=(15,80,250), clahe_clip=2.5, bilateral_d=9, unsharp_amount=1.0, saturation_mult=1.2, lap_strength=1.2),
        ParamSet(msr_sigmas=(10,60,200), clahe_clip=3.0, bilateral_d=9, unsharp_amount=1.2, saturation_mult=1.3, lap_strength=1.4),
        ParamSet(msr_sigmas=(30,120,300), clahe_clip=1.8, bilateral_d=5, unsharp_amount=0.8, saturation_mult=1.05, lap_strength=0.8),
    ]
    best_score = -1e9
    best_img = None
    best_params = None
    for p in candidate_params:
        try:
            cand = enhance_candidate(img_rgb_uint8, p)
            score = combined_natural_score(cand)
            # penalize extreme over-saturation by simple heuristic: high saturation variance
            sat = cv2.cvtColor(cand, cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32)
            sat_var = sat.std()
            # discourage extremely high sat variance
            score = score - 0.02 * max(0, sat_var - 60)
            if score > best_score:
                best_score = score
                best_img = cand
                best_params = p
        except Exception as e:
            # skip candidate on failure
            print("Candidate failed:", e)
            continue
    # final light postprocessing: gentle toning via gamma and small unsharp
    final = best_img
    # mild gamma correction toward brighter midtones
    gamma = 1.0 / 0.95
    inv_lut = np.array([((i/255.0) ** (gamma)) * 255.0 for i in range(256)]).astype(np.uint8)
    final = cv2.LUT(final, inv_lut)
    final = unsharp_mask(final, amount=0.6, radius=0.8)
    return final, best_params

# ---- Batch processing for folder pairs ----
def process_pair_folder(in_dir, out_dir, save_dir, metrics):
    os.makedirs(save_dir, exist_ok=True)
    in_paths = sorted(glob.glob(os.path.join(in_dir, "*")))
    gt_map = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "*"))):
        gt_map[os.path.basename(p)] = p
    for p in tqdm(in_paths, desc=f"Processing {os.path.basename(in_dir)}"):
        name = os.path.basename(p)
        try:
            img = load_rgb(p)
        except:
            print("Failed to load", p); continue
        img_rs = resize_rgb(img, (IMG_SIZE, IMG_SIZE))
        enhanced, chosen = auto_tune_and_enhance(img_rs)
        save_path = os.path.join(save_dir, name)
        save_rgb(save_path, enhanced)
        # compute metrics if GT exists
        if name in gt_map:
            gt = load_rgb(gt_map[name])
            gt_rs = resize_rgb(gt, (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.0
            out_f = enhanced.astype(np.float32)/255.0
            try:
                psnr = peak_signal_noise_ratio(gt_rs, out_f, data_range=1.0)
                ssim = structural_similarity(gt_rs, out_f, channel_axis=2, data_range=1.0)
            except Exception:
                psnr = float('nan'); ssim = float('nan')
        else:
            psnr = float('nan'); ssim = float('nan')
        metrics.append([name, psnr, ssim, str(chosen)])

# ---- Main pipeline ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_copy_sample", action="store_true",
                        help="If sample /mnt/data/459.jpg exists, copy it to data/test/input/")
    args = parser.parse_args()
    sample_src = "/mnt/data/459.jpg"
    if args.force_copy_sample and os.path.exists(sample_src):
        dst = os.path.join(DATA_DIR, "test", "input", os.path.basename(sample_src))
        shutil.copy(sample_src, dst)
        print("Copied sample to", dst)

    metrics = []
    total = 0
    for name, in_sub, out_sub in FOLDER_LIST:
        inp_dir = os.path.join(DATA_DIR, name, in_sub)
        out_dir = os.path.join(DATA_DIR, name, out_sub)
        save_dir = os.path.join(ENH_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        process_pair_folder(inp_dir, out_dir, save_dir, metrics)
        total += len(glob.glob(os.path.join(inp_dir, "*")))
    # write csv
    if len(metrics) > 0:
        df = pd.DataFrame(metrics, columns=["Image","PSNR","SSIM","BestParams"])
        df.to_csv(METRICS_CSV, index=False)
        print("Saved metrics:", METRICS_CSV)
    print("Done. Processed images saved in:", ENH_DIR)

if __name__ == "__main__":
    main()
