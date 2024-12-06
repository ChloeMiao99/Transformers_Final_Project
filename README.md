# Detecting Key Frames in Videos Using Vision Transformers

## Overview

### Objective
The primary goal of this project is to detect key frames in videos—frames that capture significant visual changes—using a fine-tuned Vision Transformer (ViT). These key frames serve as a summary of the video, focusing on moments that are visually distinct or meaningful, such as transitions, events, or changes in objects and scenes.

### Context
Key frame detection is a critical step in applications like:
- Video Summarization: Extracting the most important parts of a video to save time during review.
- Scene Segmentation: Dividing videos into distinct segments based on visual changes.
- Anomaly Detection: Identifying unusual events in surveillance or industrial monitoring.
- Media Analytics: Highlighting moments in sports, entertainment, or marketing videos.

Traditional methods for key frame detection rely on handcrafted features (e.g., edge detection, histogram comparisons) or pixel-wise comparisons, which are computationally expensive, inflexible, and often fail to adapt to diverse video content. This project leverages the attention mechanism of Vision Transformers to analyze high-dimensional image features and identify frames with significant changes more effectively.

### Approach
1. **Frame Extraction and Preprocessing**:
   - Videos are split into individual frames.
   - Frames are resized to 224x224 pixels to match the input requirements of the ViT model.
   - Frames are converted to RGB format and normalized for feature extraction.

2. **Model Architecture**:
   - The project uses a pre-trained Vision Transformer (`google/vit-base-patch16-224`) as the base model.
   - A fully connected layer is added to project the high-dimensional embeddings into a 128-dimensional feature space, suitable for contrastive loss training.
   - The model computes embeddings for each frame, capturing its semantic and visual features.

3. **Loss Function**:
   - The **Contrastive Loss** is used to train the model to identify differences between consecutive frames. Frames with similar features are assigned a lower loss, while those with significant differences have a higher loss.

4. **Key Frame Detection**:
   - Frame embeddings are compared using Euclidean distance.
   - If the distance exceeds a predefined threshold, the frame is considered a "key frame" and saved for further analysis.

5. **Output**:
   - Key frames are saved to a designated folder and visualized using matplotlib.
   - The feature differences are also logged, providing insights into why certain frames were chosen.


### Challenges Addressed
Traditional key frame detection often relies on handcrafted features, which are limited in their adaptability. This project leverages the power of Vision Transformers to learn robust, high-dimensional visual representations for detecting changes in a data-driven manner.

### Significance
This approach demonstrates the power of Vision Transformers in video analysis tasks. By leveraging their ability to process high-dimensional representations and focus on areas of interest, this project provides an efficient and flexible method for detecting key frames, adaptable to a wide range of applications.

## Model Card

## Critical Analysis

### Impact
The project showcases the potential of using Vision Transformers (ViT) for video analysis, specifically in tasks like key frame detection. This approach has several real-world applications:
- **Surveillance**: Automatically highlight frames where unusual activity occurs, such as the appearance of new objects or rapid scene changes.
- **Sports and Entertainment**: Identify highlights in sports matches or create video trailers by extracting visually impactful frames.
- **Medical Imaging**: Analyze video feeds from medical procedures to pinpoint critical moments for review or diagnosis.
- **Content Creation**: Assist video editors by automatically selecting significant frames for thumbnails or previews.

By reducing the amount of redundant video data while preserving critical information, this project contributes to more efficient data processing and storage.


### Challenges and Limitations 
1. **Threshold Selection**:
   - The threshold for detecting significant feature differences is crucial. A low threshold may result in too many frames being classified as "key," making the output noisy and redundant. Conversely, a high threshold might miss important changes, especially subtle transitions in videos.

2. **Computation Time**:
   - Processing videos frame-by-frame with a transformer model is computationally intensive. While effective for small datasets or offline processing, this approach may not scale well to real-time applications or large video datasets.

3. **Data Quality**:
   - The method assumes high-quality input frames with consistent resolutions and lighting. Factors like motion blur, low resolution, or varying frame rates can reduce the accuracy of feature extraction and key frame detection.

5. **Attention Visualization**:
   - Although the project identifies key frames, it does not explicitly explain which regions of the frame contributed most to the decision. Adding attention visualizations (e.g., Grad-CAM) could improve interpretability.


### Strengths
1. **Robust Feature Extraction**:
   - Vision Transformers are well-suited for analyzing complex, high-dimensional visual data, making them ideal for detecting subtle frame differences.
2. **Flexibility**:
   - The model can be fine-tuned for various applications, from sports analytics to medical imaging, with minimal modifications.
3. **Data-Driven**:
   - Unlike handcrafted methods, this approach learns meaningful features directly from data, improving adaptability to new domains.


### Future Steps
1. **Optimization**:
   - Explore lightweight transformer architectures or approximate nearest neighbor methods for faster frame comparisons.

2. **Multimodal Integration**:
   - Combine visual data with audio or text inputs (e.g., subtitles) for richer key frame detection.

3. **Explainability**:
   - Add attention map visualizations to show which parts of a frame influenced its selection as a key frame.

4. **Cross-Domain Testing**:
   - Test the model on videos from diverse domains (e.g., wildlife documentaries, industrial inspections) to assess its generalizability and fine-tune as necessary.
  
## Resources & Citations

