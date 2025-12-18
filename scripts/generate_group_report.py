import sys
import os
import torch
import numpy as np
import cv2
import time
import json
import random
import math
import csv
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

"""
Generates an advanced HTML report for colorization performance grouped by semantic categories.
Includes visual comparisons and metrics (RMSE, PSNR, Colorfulness).
"""

# Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import IMAGE_SIZE, DEVICE, CHECKPOINTS_DIR, NUM_COLOR_CLASSES, VAL_DIR, VAL_MAPPING_CSV, DATA_DIR, BASE_DIR, RESUME_FROM, T_ANNEAL
from src.model import ColorizationModel
from src.color_utils import decode_annealed_mean, rgb_to_lab, lab_to_rgb
import base64

# Configuration
GROUPS_JSON = os.path.join(DATA_DIR, "imagenet_class_groups.json")
CLASS_INDEX_JSON = os.path.join(BASE_DIR, "imagenet_class_index.json")

OUTPUT_HTML = "enhanced_report.html"
SAMPLES_PER_GROUP = 15
TEMP = T_ANNEAL
MODEL_NAME = "checkpoint_last_13_12.pth.tar" 

def img_to_base64(img_rgb, quality=70):
    """Image to base64 conversion for HTML."""
    if img_rgb.max() <= 1.0: img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_rgb)
    buff = BytesIO()
    img_pil.save(buff, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buff.getvalue()).decode('utf-8')}"


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_colorfulness(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return std_root + (0.3 * mean_root)

def generate_html(stats_by_group, detailed_results):
    print("Generating HTML")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Deep Colorization Analytics</title>
        <style>
            :root {{ --bg: #121212; --card: #1e1e1e; --text: #e0e0e0; --accent: #00d2ff; --border: #333; }}
            body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; }}
            h1, h2 {{ font-weight: 300; letter-spacing: 1px; }}
            h1 {{ text-align: center; color: var(--accent); margin-bottom: 40px; }}
            
            /* STATISTICS TABLE */
            .stats-container {{ max-width: 1200px; margin: 0 auto 50px auto; background: var(--card); border-radius: 12px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--border); }}
            th {{ color: var(--accent); font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
            tr:hover {{ background: #252525; }}
            
            /* GROUP SECTIONS */
            .group-header {{ display: flex; align-items: center; margin-top: 60px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }}
            .group-title {{ font-size: 1.5em; color: #fff; margin-right: 20px; }}
            .group-badge {{ background: #333; padding: 4px 10px; border-radius: 4px; font-size: 0.8em; color: #aaa; }}
            
            /* IMAGE CARDS */
            .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
            .card {{ background: var(--card); border-radius: 8px; overflow: hidden; transition: transform 0.2s; border: 1px solid var(--border); }}
            .card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.4); }}
            
            .img-strip {{ display: flex; height: 180px; }}
            .img-box {{ flex: 1; position: relative; overflow: hidden; }}
            .img-box img {{ width: 100%; height: 100%; object-fit: cover; }}
            .label {{ position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); color: #fff; font-size: 10px; padding: 4px; text-align: center; }}
            
            .meta {{ padding: 15px; font-size: 0.9em; }}
            .meta-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
            .metric-label {{ color: #888; }}
            .metric-val {{ font-weight: bold; }}
            
            /* LIGHTBOX */
            .lightbox {{ display: none; position: fixed; z-index: 999; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.9); justify-content: center; align-items: center; }}
            .lightbox-content {{ position: relative; width: 90%; max-width: 1200px; margin: auto; display: flex; flex-direction: column; align-items: center; }}
            .lightbox-strip {{ display: flex; width: 100%; gap: 10px; margin-bottom: 20px; }}
            .lightbox-box {{ flex: 1; }}
            .lightbox-box img {{ width: 100%; height: auto; border-radius: 4px; }}
            .lightbox-box h3 {{ color: var(--accent); text-align: center; margin: 10px 0; font-size: 1.2em; }}
            .close {{ position: absolute; top: -40px; right: 0; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; transition: 0.3s; }}
            .close:hover {{ color: var(--accent); }}
        </style>
    </head>
    <body>
        <h1>Semantic Colorization Report (T={TEMP})</h1>

        <div class="stats-container">
            <h2>Performance by Category</h2>
            <table>
                <thead>
                    <tr>
                        <th>Semantic Group</th>
                        <th>Samples</th>
                        <th>Avg RMSE (Error)</th>
                        <th>Avg PSNR (Quality)</th>
                        <th>Colorfulness Ratio</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Generating table rows
    for group in sorted(stats_by_group.keys()):
        stats = stats_by_group[group]
        if stats['count'] == 0: continue
        avg_rmse = stats['rmse_sum'] / stats['count']
        avg_psnr = stats['psnr_sum'] / stats['count']
        c_ratio = (stats['cf_pred_sum'] / stats['count']) / ((stats['cf_gt_sum'] / stats['count']) + 1e-6)
        
        verdict = "OK"
        v_color = "good"
        if c_ratio < 0.7: 
            verdict = "Gray/Sepia"
            v_color = "bad"
        elif c_ratio > 1.3:
            verdict = "Hallucinations"
            v_color = "warn"
        elif avg_rmse > 18:
            verdict = "High Error"
            v_color = "bad"
            
        html += f"""
            <tr>
                <td>{group}</td>
                <td>{stats['count']}</td>
                <td>{avg_rmse:.2f}</td>
                <td>{avg_psnr:.2f} dB</td>
                <td style="color: {'#f87171' if abs(1-c_ratio)>0.3 else '#4ade80'}">{c_ratio:.2f}</td>
                <td style="color: var(--{v_color})">{verdict}</td>
            </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    """

    # Detailed cards
    for group in sorted(detailed_results.keys()):
        items = detailed_results[group]
        if not items: continue
        html += f"""
        <div class="group-header">
            <span class="group-title">{group}</span>
            <span class="group-badge">{len(items)} samples</span>
        </div>
        <div class="gallery">
        """
        
        for item in items:
            rmse_cls = "good" if item['rmse'] < 15 else ("warn" if item['rmse'] < 20 else "bad")
            
            html += f"""
            <div class="card" onclick="openLightbox(this)">
                <div class="img-strip">
                    <div class="img-box">
                        <img src="{item['bw']}" class="src-bw">
                        <div class="label">Input</div>
                    </div>
                    <div class="img-box">
                        <img src="{item['pred']}" class="src-pred">
                        <div class="label">Prediction</div>
                    </div>
                    <div class="img-box">
                        <img src="{item['gt']}" class="src-gt">
                        <div class="label">Original</div>
                    </div>
                </div>
                <div class="meta">
                    <div class="meta-row" style="margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px;">
                        <span style="color: #fff; font-size: 0.85em; opacity: 0.7">{item['class_name']}</span>
                    </div>
                    <div class="meta-row">
                        <span class="metric-label">RMSE (ab):</span>
                        <span class="metric-val {rmse_cls}">{item['rmse']:.1f}</span>
                    </div>
                    <div class="meta-row">
                        <span class="metric-label">PSNR:</span>
                        <span class="metric-val">{item['psnr']:.1f} dB</span>
                    </div>
                    <div class="meta-row">
                        <span class="metric-label">Saturation:</span>
                        <span class="metric-val" style="color:{'#f87171' if abs(1-item['ratio'])>0.25 else '#4ade80'}">
                            {item['ratio']:.2f}x
                        </span>
                    </div>
                </div>
            </div>
            """
        html += "</div>"
    
    # Lightbox Modal & Script
    html += """
    <!-- The Modal -->
    <div id="myLightbox" class="lightbox">
      <div class="lightbox-content">
        <span class="close" onclick="closeLightbox()">&times;</span>
        <h2 id="lb-title" style="color:white; margin-bottom:20px;">Comparison</h2>
        <div class="lightbox-strip">
            <div class="lightbox-box">
                <h3>Input</h3>
                <img id="lb-bw" src="">
            </div>
            <div class="lightbox-box">
                <h3>Prediction</h3>
                <img id="lb-pred" src="">
            </div>
             <div class="lightbox-box">
                <h3>Original</h3>
                <img id="lb-gt" src="">
            </div>
        </div>
      </div>
    </div>

    <script>
        var modal = document.getElementById("myLightbox");
        
        function openLightbox(card) {
            var bwSrc = card.querySelector('.src-bw').src;
            var predSrc = card.querySelector('.src-pred').src;
            var gtSrc = card.querySelector('.src-gt').src;
            var title = card.querySelector('.meta-row span:first-child').innerText;
            
            document.getElementById("lb-bw").src = bwSrc;
            document.getElementById("lb-pred").src = predSrc;
            document.getElementById("lb-gt").src = gtSrc;
            document.getElementById("lb-title").innerText = title;
            
            modal.style.display = "flex";
        }

        function closeLightbox() {
            modal.style.display = "none";
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                closeLightbox();
            }
        }
    </script>
    </body></html>
    """
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[OK] Report generated: {os.path.abspath(OUTPUT_HTML)}")

def main():
    if not os.path.exists(GROUPS_JSON):
        print(f"Missing JSON file: {GROUPS_JSON}. Run scripts/create_class_groups.py first!")
        return
    
    if not os.path.exists(VAL_MAPPING_CSV):
        print(f"Missing CSV file: {VAL_MAPPING_CSV}")
        return
        
    print("Loading metadata")
    with open(GROUPS_JSON, "r") as f:
        groups_map = json.load(f) # "Animals": [0, 1, 2...]
        
    with open(CLASS_INDEX_JSON, "r") as f:
        class_index = json.load(f) # "0": ["n01440764", "tench"]

    # Invert groups map: Class Index -> Group Name
    idx_to_group = {}
    for group, indices in groups_map.items():
        for idx in indices:
            idx_to_group[int(idx)] = group

    # WNID to Index
    wnid_to_idx = {}
    for idx_str, (wnid, name) in class_index.items():
        wnid_to_idx[wnid] = int(idx_str)

    # Parse Validation CSV
    val_files_by_group = defaultdict(list)
    
    print(f"Parsing {VAL_MAPPING_CSV}")
    try:
        with open(VAL_MAPPING_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader) # header
            for row in reader:
                img_id = row[0]
                pred_str = row[1]
                wnid = pred_str.split()[0]
                
                if wnid in wnid_to_idx:
                    idx = wnid_to_idx[wnid]
                    if idx in idx_to_group:
                        group = idx_to_group[idx]
                        file_name = f"{img_id}.JPEG"
                        class_name = class_index[str(idx)][1]
                        val_files_by_group[group].append({
                            "path": os.path.join(VAL_DIR, file_name),
                            "class_name": class_name
                        })
    except Exception as e:
        print(f"CSV parsing error: {e}")
        return

    # Load Model
    ckpt_path = os.path.join(CHECKPOINTS_DIR, MODEL_NAME)
    
    if not os.path.exists(ckpt_path):
        print(f"{MODEL_NAME} not found, looking for last one")
        ckpts = [os.path.join(CHECKPOINTS_DIR, f) for f in os.listdir(CHECKPOINTS_DIR) if f.endswith(".pth.tar")]
        if ckpts:
            ckpts.sort(key=os.path.getmtime)
            ckpt_path = ckpts[-1]
            
    if not ckpt_path or not os.path.exists(ckpt_path):
         print("No checkpoint")
         return
    
    print(f"Loading model: {ckpt_path}")
    model = ColorizationModel(num_classes=NUM_COLOR_CLASSES).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Analysis
    detailed_results = defaultdict(list)
    group_stats = defaultdict(lambda: {'count': 0, 'rmse_sum': 0, 'psnr_sum': 0, 'cf_pred_sum': 0, 'cf_gt_sum': 0})
    
    print(f"Group analysis (Temp: {TEMP})")
    
    for group in sorted(groups_map.keys()):
        candidates = val_files_by_group[group]
        if not candidates:
            continue
            
        # Select samples
        random.shuffle(candidates)
        selected = candidates[:SAMPLES_PER_GROUP]
        
        for item in tqdm(selected, desc=group):
            img_path = item["path"]
            class_name = item["class_name"]
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # Loading
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # High Res Display Logic
                # Limit max dimension to 800px to save HTML space
                MAX_DIM = 800
                h, w = img_rgb_orig.shape[:2]
                scale = MAX_DIM / max(h, w)
                if scale < 1.0:
                    img_display = cv2.resize(img_rgb_orig, (0,0), fx=scale, fy=scale)
                else:
                    img_display = img_rgb_orig
                    
                # Model Input (Strict 224x224)
                img_model_input = cv2.resize(img_rgb_orig, (IMAGE_SIZE, IMAGE_SIZE))
                img_input_float = img_model_input.astype(np.float32) / 255.0

                # Inference Pipeline
                lab_input_small = rgb_to_lab(img_input_float)
                L_tensor = torch.from_numpy(lab_input_small[:,:,0] / 100.0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(L_tensor)
                    ab_dec = decode_annealed_mean(logits, T=TEMP)
                    ab_pred_small = ab_dec.squeeze(0).permute(1, 2, 0).cpu().numpy() # 112x112x2

                # High-Res Reconstruction
                # Use L from DISPLAY image + Upscaled AB from PREDICTION
                lab_display = rgb_to_lab(img_display.astype(np.float32) / 255.0)
                L_display = lab_display[:,:,0]
                
                # Resize predicted AB to match Display dimensions
                ab_pred_upscaled = cv2.resize(ab_pred_small, (img_display.shape[1], img_display.shape[0]))
                
                # Combine
                lab_pred_full = np.dstack((L_display, ab_pred_upscaled))
                img_pred_full = lab_to_rgb(lab_pred_full).clip(0, 1) # RGB Result
                
                # Originals for Comparison (also High Res)
                img_display_float = img_display.astype(np.float32) / 255.0
                img_gray_full = np.stack([L_display/100.0]*3, axis=-1).clip(0, 1)

                # Metric calculation
                
                # RMSE (AB Color Error)
                ab_gt_full = lab_display[:,:,1:]
                rmse = np.sqrt(np.mean((ab_pred_upscaled - ab_gt_full)**2))
                
                # PSNR (RGB Quality)
                psnr = calculate_psnr(img_pred_full, img_display_float)
                
                # Colorfulness (Saturation)
                cf_p = calculate_colorfulness(img_pred_full)
                cf_g = calculate_colorfulness(img_display_float)
                ratio = cf_p / (cf_g + 1e-6)

                # Saving statistics
                group_stats[group]['count'] += 1
                group_stats[group]['rmse_sum'] += rmse
                group_stats[group]['psnr_sum'] += psnr
                group_stats[group]['cf_pred_sum'] += cf_p
                group_stats[group]['cf_gt_sum'] += cf_g

                # Saving details (to cards) - NOW HIGH RESOLUTION
                detailed_results[group].append({
                    'class_name': class_name,
                    'bw': img_to_base64(img_gray_full),
                    'pred': img_to_base64(img_pred_full),
                    'gt': img_to_base64(img_display_float),
                    'rmse': rmse,
                    'psnr': psnr,
                    'ratio': ratio
                })
                
            except Exception as e:
                print(f"Error at {class_name}: {e}")
                continue

    generate_html(group_stats, detailed_results)

if __name__ == "__main__":
    main()