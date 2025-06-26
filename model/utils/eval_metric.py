import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def get_matched_image_pairs(result_dir, gt_dir, suffix='eval'):
    """
    Matches *_result.jpg files in result_dir with gt_*.jpg in gt_dir
    by shared base name.
    """
    result_files = sorted([
        f for f in os.listdir(result_dir)
        if f.endswith('_result.jpg') and f.startswith(suffix)
    ])

    matched_pairs = []
    for res_file in result_files:
        base_name = res_file.replace('_result.jpg', '')
        gt_file = f"gt_{base_name}.jpg"
        gt_path = os.path.join(gt_dir, gt_file)

        if os.path.exists(gt_path):
            matched_pairs.append((os.path.join(result_dir, res_file), gt_path))
        else:
            print(f"⚠️ GT not found for {res_file}, skipping.")

    return matched_pairs


def resize_to_match(src, target_shape):
    """Resize src image to target shape using bicubic interpolation."""
    return cv2.resize(src, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)


def evaluate_psnr(result_dir, gt_dir, suffix='eval'):
    psnr_scores = []
    pairs = get_matched_image_pairs(result_dir, gt_dir, suffix)

    for res_path, gt_path in tqdm(pairs, desc="Evaluating PSNR"):
        res_img = cv2.cvtColor(cv2.imread(res_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if res_img.shape != gt_img.shape:
            res_img = resize_to_match(res_img, gt_img.shape[:2])

        psnr = compare_psnr(gt_img, res_img, data_range=1.0)
        psnr_scores.append(psnr)

    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    print(f"\n✅ PSNR Evaluation Complete - Avg: {avg_psnr:.4f}")
    return avg_psnr


def evaluate_ssim(result_dir, gt_dir, suffix='eval'):
    ssim_scores = []
    pairs = get_matched_image_pairs(result_dir, gt_dir, suffix)

    for res_path, gt_path in tqdm(pairs, desc="Evaluating SSIM"):
        res_img = cv2.cvtColor(cv2.imread(res_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if res_img.shape != gt_img.shape:
            res_img = resize_to_match(res_img, gt_img.shape[:2])

        ssim = compare_ssim(gt_img, res_img, data_range=1.0, channel_axis=-1)
        ssim_scores.append(ssim)

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    print(f"\n✅ SSIM Evaluation Complete - Avg: {avg_ssim:.4f}")
    return avg_ssim


def evaluate_metric(metric_name, result_dir, gt_dir, suffix='eval'):
    """
    Evaluates PSNR or SSIM using resized result image matched to GT resolution.

    Args:
        metric_name (str): 'PSNR' or 'SSIM'
        result_dir (str): Folder containing *_result.jpg images
        gt_dir (str): Folder containing gt_*.jpg images
        suffix (str): Common naming prefix (e.g., 'eval')

    Returns:
        float: average metric score
    """
    metric_name = metric_name.upper()

    if metric_name == 'PSNR':
        return evaluate_psnr(result_dir, gt_dir, suffix=suffix)
    elif metric_name == 'SSIM':
        return evaluate_ssim(result_dir, gt_dir, suffix=suffix)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}. Choose 'PSNR' or 'SSIM'.")
