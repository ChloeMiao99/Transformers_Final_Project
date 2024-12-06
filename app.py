import streamlit as st
from vit_change_detector import VitChangeDetector
import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from scipy.ndimage import label

class BaselineFrameExtractor:
    """
    A class to extract and process significant frames from a video based on color histogram percentage,
    Structural Similarity Index (SSIM) thresholds, and optical flow (for detecting horizontal/vertical scrolling).
    """
    def __init__(self, video_path, output_folder, overall_ssim_threshold=0.998, batch_ssim_threshold=0.7, cluster_size=3,
                 color_pixel_threshold=0.7, num_color_bins=128, grid_size=(60, 60), movement_threshold=1, interval_seconds=0.15, video_name=None):
        """
        Initializes the FrameExtractor class with the specified parameters.
        """
        self.video_path = video_path
        self.video_name = video_name if video_name else os.path.basename(video_path).replace(" ", "-")
        self.output_folder = output_folder

        # Thresholds for SSIM and color pixel percentage
        self.ssim_threshold = overall_ssim_threshold
        self.batch_ssim_threshold = batch_ssim_threshold
        self.color_pixel_threshold = color_pixel_threshold
        # Divide the 256 pixel value into 128 bins
        self.num_color_bins = num_color_bins
        self.movement_threshold = movement_threshold

        # Grid Batch Init
        self.grid_size = grid_size
        self.cluster_size = cluster_size
        self.interval_seconds = interval_seconds

        self.extracted_timeframe = []  # Stores the timestamps of frames saved
        self.extracted_image_spec = []  # Stores list of cv2.imwrite() results

    def _clear_image_files(self, folder_path):
        """
        Deletes image files with specific extensions in the given folder.
        """
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
        for ext in image_extensions:
            files = glob.glob(os.path.join(folder_path, ext))
            for file in files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        print("Successfully removed image files")

    def _calculate_ssim(self, image1, image2):
        """
        Calculates the Structural Similarity Index (SSIM) between two images.
        """
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(gray_image1, gray_image2, full=True)
        return score, diff

    def _calculate_color_histogram_percentage(self, image, threshold=0.7):
        """
        Calculates the color histogram for the image and checks if any color exceeds the threshold percentage.
        """
        # Convert the image to HSV color space for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate the histogram for the Hue channel
        hist = cv2.calcHist([hsv_image], [0], None, [self.num_color_bins], [0, 256])

        # Normalize the histogram to get percentages
        hist_norm = hist / hist.sum()

        # Check if any color bin exceeds the threshold
        if (hist_norm > threshold).any():
            return True  # If any color dominates, return True (frame considered insignificant)
        return False  # Otherwise, return False (frame is significant)

    def _crop_image_into_grid(self, image):
        """
        Crops the input image into smaller sections using numpy based on the specified grid size.

        Args:
            image (ndarray): The input image to be cropped.


        Returns:
            ndarray: A numpy array containing the cropped sections of the image. Each section is
                    of size (crop_height, crop_width), where the image is divided according to
                    the grid dimensions.

        Notes:
            - The image is resized to ensure it is divisible by the grid dimensions. Excess pixels
            that do not fit into the grid are discarded.
        """
        height, width = image.shape[:2]
        crop_height = height // self.grid_size[0]
        crop_width = width // self.grid_size[1]

        # Resize the image to ensure divisibility by grid dimensions
        resized_image = image[:crop_height * self.grid_size[0], :crop_width * self.grid_size[1]]

        # Reshape the image into grid sections
        cropped_images = resized_image.reshape(self.grid_size[0], crop_height, self.grid_size[1], crop_width).swapaxes(1, 2)
        return cropped_images

    def _calculate_ssim_w_grid(self, image1, image2):
        """
        Computes the Structural Similarity Index (SSIM) over cropped sections of two images using numpy.

        Args:
            image1 (ndarray): The first image for SSIM comparison.
            image2 (ndarray): The second image for SSIM comparison.
            grid_size (tuple, optional): The grid size specifying how many sections (rows, columns)
                                        to divide the image into for localized SSIM comparison.
                                        Defaults to (5, 5).

        Returns:
            ndarray: A matrix of SSIM scores where each element corresponds to the SSIM score of
                    the corresponding cropped section of the two images.

        Notes:
            - The images are converted to grayscale before SSIM calculation.
            - The SSIM score for each corresponding section of the two images is computed and stored
            in an SSIM scores matrix.
        """
        # Convert images to grayscale
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Crop images into smaller sections using numpy
        cropped_images1 = self._crop_image_into_grid(gray_image1)
        cropped_images2 = self._crop_image_into_grid(gray_image2)

        # Initialize an empty matrix to hold SSIM scores
        ssim_scores_matrix = np.zeros(self.grid_size)

        # Compare each corresponding section and store SSIM in matrix
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                score, _ = ssim(cropped_images1[i, j], cropped_images2[i, j], full=True)
                ssim_scores_matrix[i, j] = score

        return ssim_scores_matrix

    def extract_and_save_significant_frames(self):
        print("Loading the videos")
        video = cv2.VideoCapture(self.video_path)

        if not video.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            frame_interval = max(1, int(fps * self.interval_seconds))
        else:
            print("Error: FPS could not be retrieved.")
            return

        video_output_folder = os.path.join(self.output_folder, self.video_name)
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)

        frame_count = 0
        saved_frame_count = 0
        last_saved_frame = None

        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                if last_saved_frame is None:
                    timestamp = frame_count / fps
                    self.extracted_image_spec.append(frame)
                    self.extracted_timeframe.append(timestamp)
                    last_saved_frame = frame
                    saved_frame_count += 1
                else:
                    if not self._calculate_color_histogram_percentage(frame, threshold=self.color_pixel_threshold):
                        ssim_score, _ = self._calculate_ssim(last_saved_frame, frame)
                        if ssim_score < self.ssim_threshold:
                            timestamp = frame_count / fps
                            self.extracted_image_spec.append(frame)
                            self.extracted_timeframe.append(timestamp)
                            last_saved_frame = frame
                            saved_frame_count += 1

            frame_count += 1

        video.release()
        print(f"Extraction complete: {saved_frame_count} frames saved.")

        # Perform batch SSIM processing here
        print("Processing Batch SSIM...")
        self.process_frames_w_ssim_batch()

        return self.extracted_timeframe, self.extracted_image_spec

    def process_frames_w_ssim_batch(self):
        if not self.extracted_image_spec and not self.extracted_timeframe:
            self.extract_and_save_significant_frames()
            print("Extracting frames from video")

        print("Original Before batch SSIM Frame")
        last_saved_frame = self.extracted_image_spec[0]
        extracted_frame = [last_saved_frame]
        last_saved_timeframe = self.extracted_timeframe[0]
        extracted_timeframe = [self.extracted_timeframe[0]]

        i = 1
        while i < len(self.extracted_image_spec):
            new_frame = self.extracted_image_spec[i]
            new_timeframe = self.extracted_timeframe[i]
            ssim_scores = self._calculate_ssim_w_grid(last_saved_frame, new_frame)

            boulder_mask = ssim_scores < self.batch_ssim_threshold
            arr_converted = boulder_mask.astype(int)

            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            labeled_matrix, num_features = label(arr_converted, structure=structure)

            cluster_sizes = np.bincount(labeled_matrix.ravel())[1:]

            if np.any(cluster_sizes > self.cluster_size):
                last_saved_frame = new_frame
                last_saved_timeframe = new_timeframe
                extracted_timeframe.append(new_timeframe)
                extracted_frame.append(new_frame)

            i += 1

        self.extracted_timeframe = extracted_timeframe
        self.extracted_image_spec = extracted_frame
        print("Original After batch SSIM Frame")
        print(f"Len: {len(self.extracted_timeframe)}")
        return self.extracted_image_spec

    def _calculate_optical_flow(self, image1, image2):
        """
        Calculates the optical flow between two images and returns the direction of movement (Horizontal or Vertical).
        """
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using the Farneback method
        flow = cv2.calcOpticalFlowFarneback(gray_image1, gray_image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract the x-component (horizontal) and y-component (vertical) of the flow
        horizontal_flow = flow[..., 0]
        vertical_flow = flow[..., 1]

        # Calculate the average magnitude of horizontal and vertical flow
        avg_horizontal_flow = np.mean(np.abs(horizontal_flow))
        avg_vertical_flow = np.mean(np.abs(vertical_flow))

        # Return the higher flow value and direction
        if avg_horizontal_flow > avg_vertical_flow:
            return avg_horizontal_flow, "Horizontal"
        else:
            return avg_vertical_flow, "Vertical"

    def remove_horizontal_optical_movement(self):
        """
        Processes frames and removes those where horizontal movement (scrolling) is detected.
        """
        # Check if there are frames to process
        if not self.extracted_image_spec and not self.extracted_timeframe:
            self.extract_and_save_significant_frames()
            print("Extracting frames from video")

        print("Processing Frames for Horizontal/Vertical Movement")

        # Initialize dictionary to store frame information
        frame_time = self.extracted_timeframe
        image_imread = self.extracted_image_spec
        all_frames = {frame_time[i]: image_imread[i] for i in range(len(frame_time))}

        # Initialize a list to store movement data
        movement_data = []
        previous_image = image_imread[0]

        # Iterate through the images to compare movement
        for i in range(1, len(image_imread)):
            current_image = image_imread[i]
            current_image_timeframe = frame_time[i]
            avg_flow, label_flow = self._calculate_optical_flow(previous_image, current_image)

            # Determine if there is significant movement based on the threshold
            has_movement = avg_flow > self.movement_threshold

            # Append the result to movement data
            movement_data.append({
                'frame1': frame_time[i - 1],
                'frame2': frame_time[i],
                'movement_score': avg_flow,
                'movement_label': label_flow,
                'has_movement': has_movement
            })

            previous_image = current_image

        # Convert the movement data into a pandas DataFrame
        df = pd.DataFrame(movement_data)

        # Apply sandwich logic for filtering frames with movement
        df['new_logic'] = self.apply_sandwich_logic_with_skip(df)

        # List of frames to retain (those without significant movement)
        image_timeframe_list = []
        consecutive_true = False

        for i, row in df.iterrows():
            if not row['has_movement']:
                image_timeframe_list.append(row['frame1'])
            else:
                if not consecutive_true:
                    image_timeframe_list.append(row['frame1'])
                    consecutive_true = True
            if not row['has_movement']:
                consecutive_true = False

        # Rebuild the extracted frames based on movement analysis
        matched_frames = {key: all_frames[key] for key in image_timeframe_list if key in all_frames}
        self.extracted_timeframe = list(matched_frames.keys())
        self.extracted_image_spec = list(matched_frames.values())

        return self.extracted_timeframe, self.extracted_image_spec

    def apply_sandwich_logic_with_skip(self, df):
        """
        Implements logic to handle 'sandwiched' frames based on movement detection.
        """
        new_logic = df['has_movement'].copy()
        i = 0

        while i < len(df) - 1:
            if (df['has_movement'].iloc[i] == False) and (df['has_movement'].iloc[i + 1] == True) and \
                    (i + 2 < len(df)) and (df['has_movement'].iloc[i + 2] == False):
                new_logic.iloc[i + 1] = True
                i += 2
            else:
                i += 1

        return new_logic

    def generate_frames(self):
        """
        Saves the extracted frames to the output folder.
        """
        video_output_folder = os.path.join(self.output_folder, self.video_name)
        self._clear_image_files(video_output_folder)
        saved_frame_count = 0

        for i in range(len(self.extracted_timeframe)):
            frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:05d}_time_{self.extracted_timeframe[i]:.2f}.png")
            cv2.imwrite(frame_filename, self.extracted_image_spec[i])
            saved_frame_count += 1

        print("Finished generating frames.")

def process_video_streamlit(video_path, output_folder, threshold=0.5):
    """
    Process a video through SSIM and ViT detection to extract significant frames.

    Args:
        video_path (str): Path to input video file.
        output_folder (str): Directory where frames will be saved.
        threshold (float): Sensitivity threshold for ViT change detection.

    Returns:
        tuple: (timeframe_list, frame_dict) where:
            - timeframe_list: List of timestamps for significant frames.
            - frame_dict: Dictionary mapping timestamps to frame images.
    """
    baseline_detector = BaselineFrameExtractor(
        video_path=video_path,
        output_folder=output_folder
    )

    timeframe_list, image_spec_list = baseline_detector.remove_horizontal_optical_movement()

    vit_detector = VitChangeDetector(threshold=threshold)

    final_timeframes, final_frames_dict = vit_detector.process_frames(
        timeframe_list,
        image_spec_list
    )

    final_output_folder = os.path.join(output_folder, 'final_frames')
    vit_detector.generate_frames(
        final_timeframes,
        final_frames_dict,
        final_output_folder
    )

    return final_timeframes, final_frames_dict, final_output_folder

# Streamlit App
st.title("Key Frame Extraction App")

st.write("""
This app allows you to upload a video and extract the key frames using 
Vision Transformer (ViT) for change detection.
""")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    video_path = f"temp_{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Set output folder
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)

    st.write("Processing video...")
    with st.spinner("Extracting key frames..."):
        timeframes, frames_dict, final_output_folder = process_video_streamlit(
            video_path=video_path,
            output_folder=output_folder,
            threshold=0.3
        )

    st.success("Processing complete!")

    # Display extracted frames
    st.write("### Extracted Key Frames")
    for timestamp, frame in frames_dict.items():
        # Convert frame (numpy array) to PIL Image for display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(img, caption=f"Frame at {timestamp:.2f} seconds", use_column_width=True)

    # Provide download link for frames
    st.write("### Download Extracted Frames")
    zip_file = os.path.join(output_folder, "final_frames.zip")
    os.system(f"zip -r {zip_file} {final_output_folder}")

    with open(zip_file, "rb") as f:
        st.download_button(
            label="Download Frames as ZIP",
            data=f,
            file_name="key_frames.zip",
            mime="application/zip"
        )