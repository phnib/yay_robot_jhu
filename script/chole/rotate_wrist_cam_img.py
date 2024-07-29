import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def get_image_paths(base_dir):
    """Retrieve all image paths from the given base directory."""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        # for file in files:
        file = random.choice(files)
        if file.endswith('.jpg'):
            image_paths.append(os.path.join(root, file))
    return image_paths

def find_rotation_angle(image):
    """Find the rotation angle to rectify the image and visualize the steps."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute the minimum area rectangle around the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Compute the angle of the rectangle
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Visualization of the processing steps
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Grayscale Image')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Thresholded Image')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Contours and Min Area Rect')
    plt.imshow(image)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    plt.axis('off')
    
    plt.show()
    
    return angle


def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def draw_center_line(image):
    """Draw a vertical center line on the image."""
    (h, w) = image.shape[:2]
    center_x = w // 2
    cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 0), 2)
    return image

def shift_image(image, shift_x, shift_y):
    """Shift the image by the given x and y offsets."""
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image

def process_images(image_paths):
    """Process images from the given directory."""
    # image_paths = get_image_paths(base_dir)
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        # Read the image
        image = cv2.imread(image_path)
        original_image = image.copy()

        # Find the rotation angle
        # angle = find_rotation_angle(image)
        angle = -52.0

        # Rotate the image
        rotated_image = rotate_image(image, angle)

        # Shift the image (adjust shift_x and shift_y as needed)
        shift_x, shift_y = 10, 0 
        shifted_image = shift_image(rotated_image, shift_x, shift_y)
        # Draw the center line on the rotated image
        rotated_image_with_line = draw_center_line(shifted_image)
        
        # Visualize the original and rectified images
        visualize_images(original_image, rotated_image_with_line, image_path, angle)


def visualize_images(original, rectified, path, angle):
    """Visualize the original and rectified images."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Rectified (Rotated by {angle:.2f} degrees)')
    plt.imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.suptitle(f'Image: {os.path.basename(path)}')
    plt.show()

if __name__ == "__main__":
    base_dir = '/home/imerse/chole_ws/data/base_chole_clipping_cutting'
    tissue_ids = [5, 6, 8, 12, 13, 14, 18]

    for tissue_id in tissue_ids:
        # dataset_path = f"/home/imerse/chole_ws/data/phantom_chole/phantom_{tissue_id}/{phase}"
        dataset_path = f"/home/imerse/chole_ws/data/base_chole_clipping_cutting/tissue_{tissue_id}"
        phases = os.listdir(dataset_path)
        # print(phases)
        phase = random.choice(phases)
        # for phase in phases:
        samples = os.listdir(os.path.join(dataset_path, phase))
        # print(samples)
        sample = random.choice(samples)

        # for sample in samples:
        sample_dir = os.path.join(dataset_path, phase, sample)
        if sample == "Corrections":
            s = os.listdir(sample_dir)
            for ss in s:
                sample_dir = os.path.join(sample_dir, ss)
                break

        # if not os.path.exists(os.path.join(sample_dir, "ee_csv.csv")):
        #     print(f"ee state csv file not found in {sample_dir}")
        #     exit

        psm1_img_paths = os.path.join(sample_dir, "endo_psm1")
        image_paths = get_image_paths(psm1_img_paths)
        # print(image_paths)
        process_images(image_paths)
        # input("enter to continue...")

