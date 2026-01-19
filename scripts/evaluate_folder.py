import sys
import os
import glob
import argparse
import torch
import numpy as np
import cv2
import lpips
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ColorizationModel
from src.color_utils import decode_annealed_mean, lab_to_rgb, rgb_to_lab
from configs.config import DEVICE, CHECKPOINTS_DIR, NUM_COLOR_CLASSES, T_ANNEAL, IMAGE_SIZE

def calculate_colorfulness(img_rgb):
    if img_rgb.max() <= 1.0:
        img = (img_rgb * 255).astype(np.uint8)
    else:
        img = img_rgb.astype(np.uint8)
        
    R, G, B = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return std_root + (0.3 * mean_root)

def load_model(device, checkpoint_path=None):
    if not checkpoint_path:
        if not os.path.exists(CHECKPOINTS_DIR):
            return None
        ckpts = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "*.ckpt")), key=os.path.getmtime)
        if not ckpts:
            return None
        checkpoint_path = ckpts[-1]
    
    print(f"Loading model: {checkpoint_path}")
    model = ColorizationModel(num_classes=NUM_COLOR_CLASSES).to(device)
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        new_state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
    except Exception:
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    model.eval()
    return model

def process_image(model, img_path, device, loss_fn_lpips):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return
    
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb_orig.shape[:2]
    
    img_resized = cv2.resize(img_rgb_orig, (IMAGE_SIZE, IMAGE_SIZE))
    img_float = img_resized.astype(np.float32) / 255.0
    lab_input = rgb_to_lab(img_float)
    L_tensor = torch.from_numpy(lab_input[:,:,0] / 100.0).float().unsqueeze(0).unsqueeze(0).to(device)
    
    logits = model(L_tensor)
    ab_pred = decode_annealed_mean(logits, T=T_ANNEAL).squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab_pred_upscaled = cv2.resize(ab_pred, (w, h))
    
    L_orig = rgb_to_lab(img_rgb_orig.astype(np.float32) / 255.0)[:,:,0]
    lab_pred = np.dstack((L_orig, ab_pred_upscaled))
    rgb_pred = lab_to_rgb(lab_pred).clip(0, 1)
    
    t_pred = torch.from_numpy(rgb_pred).float().permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    t_gt = torch.from_numpy(img_rgb_orig.astype(np.float32) / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    
    lpips_val = loss_fn_lpips(t_pred, t_gt).item()
    ssim_val = ssim(img_rgb_orig.astype(np.float32)/255.0, rgb_pred, data_range=1.0, channel_axis=2)
    
    cf_pred = calculate_colorfulness(rgb_pred)
    cf_orig = calculate_colorfulness(img_rgb_orig.astype(np.float32) / 255.0)
    ratio = cf_pred / (cf_orig + 1e-6)
    
    print(f"{os.path.basename(img_path):<25} | {lpips_val:.4f}     | {ssim_val:.4f}     | {ratio:.4f}       | {cf_pred:.2f}     | {cf_orig:.2f}")
    
    out_path = os.path.splitext(img_path)[0] + "_colorized.png"
    cv2.imwrite(out_path, cv2.cvtColor((rgb_pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test_images")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    model = load_model(device, args.checkpoint)
    if not model: return
    
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    
    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        return
        
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files.extend(glob.glob(os.path.join(args.folder, ext)))
        
    print(f"Found {len(files)} images in {args.folder}")
    print(f"{'Filename':<25} | {'LPIPS (v)':<10} | {'SSIM (^)':<10} | {'CF Ratio (~1)':<12} | {'CF Pred':<8} | {'CF Orig':<8}")
    print("-" * 90)
    
    with torch.no_grad():
        for f in files:
            process_image(model, f, device, loss_fn_lpips)

if __name__ == "__main__":
    main()
