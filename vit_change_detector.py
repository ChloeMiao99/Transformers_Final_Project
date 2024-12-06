import torch
from transformers import ViTFeatureExtractor, ViTModel
import cv2
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks

class VitChangeDetector:
    """"A class that uses Google's Vision Transformer (ViT) for detecting significant changes between frames.
    It combines both global (entire frame) and local (patch-level) features to understand changes."""
    def __init__(self, threshold=0.05):  
        """Initialize VIT detector"""
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224',
            do_resize=True,
            size=224
        )
        
        self.model = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            add_pooling_layer=False
        )
        self.model.eval()
        self.threshold = threshold  # Changed variable name
    
    @torch.no_grad()
    def get_frame_features(self, frame):
        """
        Extract rich features from a frame using ViT model.
        Combines global features (CLS token) and local features (patch tokens) from last 3 layers.

        Args:
            frame: Input frame (numpy array in BGR format)

        Returns:
            list: Combined features from last 3 layers of ViT model
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.feature_extractor(
            images=frame_rgb,
            return_tensors="pt"
        )
        
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get features from last 3 layers for robustness
        features = []
        for layer_output in outputs.hidden_states[-3:]:
            global_feat = layer_output[:, 0]
            local_feat = layer_output[:, 1:].mean(dim=1)
            combined = (global_feat + local_feat) / 2
            features.append(combined)
            
        return features
    
    def compute_change_score(self, features1, features2):
        """
        Compute change score between two frames using their features.
        Uses MSE loss to calculate difference between corresponding feature vectors.

        Args:
            features1: Features from first frame
            features2: Features from second frame

        Returns:
            float: Mean change score across all feature layers
        """

        scores = []
        for f1, f2 in zip(features1, features2):
            f1 = f1.float()
            f2 = f2.float()
            score = torch.nn.functional.mse_loss(f1, f2)
            scores.append(score.item())
        return np.mean(scores)
    
    def process_frames(self, timeframe_list, image_spec_list):
        """
        Process list of frames to detect significant changes using ViT features.
        Creates dictionary mapping, computes change scores, and uses peak detection
        to identify frames with significant changes.

        Args:
            timeframe_list: List of frame timestamps
            image_spec_list: List of corresponding frame images

        Returns:
            tuple: (list of selected timestamps, dictionary of selected frames)
        """
        print(f"Processing {len(timeframe_list)} frames through ViT...")
        
        # Create frame dictionary
        frame_dict = {
            timeframe_list[i]: image_spec_list[i] 
            for i in range(len(timeframe_list))
        }
        
        # Extract features and compute changes
        all_features = []
        change_scores = []
        
        print("Computing frame features...")
        for time in tqdm(timeframe_list):
            features = self.get_frame_features(frame_dict[time])
            all_features.append(features)
            
            if len(all_features) > 1:
                score = self.compute_change_score(all_features[-1], all_features[-2])
                change_scores.append(score)
                print(f"Frame {len(all_features)-1} change score: {score:.4f}")
        
        # Normalize and detect changes
        if change_scores:
            change_scores = np.array(change_scores)
            change_scores = (change_scores - np.min(change_scores)) / (np.max(change_scores) - np.min(change_scores))
            
            peaks = find_peaks(change_scores, prominence=self.threshold)[0]
            
            new_timeframe_list = [timeframe_list[0]]  # Keep first frame
            
            for peak in peaks:
                new_timeframe_list.append(timeframe_list[peak + 1])
                
            if timeframe_list[-1] not in new_timeframe_list:
                new_timeframe_list.append(timeframe_list[-1])
                
            new_timeframe_list = sorted(list(set(new_timeframe_list)))
        else:
            new_timeframe_list = timeframe_list
        
        # Create filtered dictionary
        filtered_frame_dict = {
            time: frame_dict[time] 
            for time in new_timeframe_list
        }
        
        print(f"ViT filtering complete. Kept {len(new_timeframe_list)} significant frames")
        print(f"Selected timestamps: {[f'{t:.2f}' for t in new_timeframe_list]}")
        
        return new_timeframe_list, filtered_frame_dict

    def generate_frames(self, timeframe_list, frame_dict, output_folder):
        """
        Save selected significant frames to disk.
        Generates numbered frame files with timestamps in filename.

        Args:
            timeframe_list: List of selected frame timestamps
            frame_dict: Dictionary mapping timestamps to frames
            output_folder: Directory to save frames

        Returns:
            None
        """
        os.makedirs(output_folder, exist_ok=True)
        
        for i, time in enumerate(timeframe_list):
            frame = frame_dict[time]
            filename = f"frame_{i:05d}_time_{time:.2f}.png"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            
        print(f"Generated {len(timeframe_list)} final frames in {output_folder}")