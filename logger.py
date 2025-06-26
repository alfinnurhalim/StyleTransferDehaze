import os
import torch
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

# New Logger Class
class TrainingLogger:
    def __init__(self, log_path, log_filename='training.log', graph_filename='train_loss.png'):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, log_filename)
        self.graph_path = os.path.join(log_path, graph_filename)
        self.loss_history = []
        
        # Create a logger instance
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler and set formatter
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Avoid duplicate handlers if already set
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def log_epoch_loss(self, epoch, loss):
        # Convert loss (Tensor or float) to a CPU float
        if isinstance(loss, torch.Tensor):
            cpu_loss = loss.cpu().item()
        else:
            cpu_loss = float(loss)

        self.logger.info(f"Epoch {epoch} - Last Batch Loss: {cpu_loss:.4f}")
        self.loss_history.append(cpu_loss)
    
    def update_graph(self):
        plt.figure()
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(self.graph_path)
        plt.close()
        print(f"Updated training graph saved at: {self.graph_path}")

import numpy as np
import cv2
import torch

import numpy as np
import cv2
import torch

import numpy as np
import cv2
import torch

def overlay_attention_on_image(attn_map, content_img, alpha=0.5):
    """
    Overlays attention heatmap on the content image.
    
    Args:
        attn_map (Tensor): [1, 1, H, W] or [1, H, W]
        content_img (Tensor): [3, H, W]
    Returns:
        overlayed image as np.uint8 array [H, W, 3]
    """
    # Step 1: Extract the attention map for the first image, squeeze to [H, W]
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map[0].squeeze().detach().cpu().numpy()  # [H, W]

    # Step 2: Resize to match content image
    H, W = content_img.shape[1], content_img.shape[2]
    attn_map = cv2.resize(attn_map, (W, H))

    # Step 3: Normalize to [0, 255]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-5)
    attn_map = (attn_map * 255).astype(np.uint8)  # OpenCV requires CV_8UC1

    # Step 4: Apply color map
    attn_color = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)  # [H, W, 3]

    # Step 5: Convert content image to OpenCV BGR format
    content_np = (content_img.detach().cpu().numpy() * 255).astype(np.uint8)
    content_np = np.transpose(content_np, (1, 2, 0))  # [H, W, C]
    content_np = cv2.cvtColor(content_np, cv2.COLOR_RGB2BGR)

    # Step 6: Overlay attention on content image
    overlayed = cv2.addWeighted(content_np, 1 - alpha, attn_color, alpha, 0)

    return overlayed