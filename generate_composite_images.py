import math
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def load_image(image_path):
    try:
        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)   # Load image with all channels
    except Exception as e:
        print(f"无法加载图像 {image_path}: {e}")
        return None

# Create a YOLO label file in the specified folder
def create_yolo_label(name, labels_folder):
    os.makedirs(labels_folder, exist_ok=True)
    label_path = os.path.join(labels_folder, name)
    try:
        label_file = open(label_path, "w")
        return label_file
    except Exception as e:
        print(f"无法创建标签文件 {label_path}: {e}")
        return None

# Resize an image while maintaining aspect ratio and applying a random scale
def resize_img(img, fg_h, fg_w, bg_h, bg_w, min_ratio, max_ratio):
    k = fg_h / fg_w
    resize_ratio_width = random.uniform(min_ratio, max_ratio)
    resize_ratio_height = resize_ratio_width * random.uniform(0.7 * k, 1.3 * k)

    new_width = int(bg_w * resize_ratio_width)
    new_height = int(bg_h * resize_ratio_height)
    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img

# Rotate an image by a random angle and crop it to its bounding box
def rotation(img):

    rotation_angle = random.randint(0, 360)  # Select a random rotation angle

    height, width = img.shape[:2]

    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)  # Generate the rotation matrix

    radians = math.radians(rotation_angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_matrix[0, 2] += ((bound_w / 2) - center[0])
    rotation_matrix[1, 2] += ((bound_h / 2) - center[1])

    rotated_img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))

    coords = find_opaque_coords(rotated_img)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    img_cropped = rotated_img[y_min:y_max, x_min:x_max]
    return img_cropped


# Find the coordinates of non-transparent pixels in the image
def find_opaque_coords(img):
    coords = np.argwhere(img[:, :, 3] > 0)
    return coords

def composite_image(coords, fg, bg, position, mask_image):
    fg_alpha = fg[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
    for coord in coords:
        y, x = coord
        fg_pixel = fg[y, x, :3]  # Extract RGB values of the foreground
        alpha = fg_alpha[y, x]
        if alpha > 0:  # Only blend if alpha > 0
            bg_pixel = bg[y+position[1], x+position[0], :]
            blended_pixel = alpha * fg_pixel + (1 - alpha) * bg_pixel
            bg[y+position[1], x+position[0], :] = blended_pixel
            mask_image[y + position[1], x + position[0]] = 255
    return bg, mask_image


# Write YOLO-format bounding box information to the label file
def write_yolo_labels(position, label_file, fg_h, fg_w, bg_h, bg_w):
    center = (position[0] + 0.5 * fg_w, position[1] + 0.5 * fg_h) # Compute the center of the bounding box

    # Convert dimensions and position to normalized values
    center_ratio = (round(center[0] / bg_w, 6), round(center[1] / bg_h, 6))
    width_and_height_ratio = (
        round(fg_w / bg_w, 6), round(fg_h / bg_h, 6))

    # Write the label data to the file
    label_file.write(
        f"0 {center_ratio[0]} {center_ratio[1]} {width_and_height_ratio[0]} {width_and_height_ratio[1]}\n")

if __name__ == '__main__':
    # 前景文件夹路径
    foreground_folder = r""
    # 背景文件夹路径
    background_folder = r""
    # 合成图片保存路径
    output_folder = r""
    # 掩码图像保存路径
    mask_folder = r""
    # Yolo标签保存路径
    labels_folder = r""
    # 每张背景贴图的数量
    num_composites = 3  # Number of foreground images to overlay per background
    # 最小缩放倍数，最大缩放倍数
    min_ratio, max_ratio = 1 / 20, 1 / 10  # Minimum and maximum scaling ratios

    print("正在加载前景图像...")
    foreground_images = {filename: load_image(os.path.join(foreground_folder, filename)) for filename in tqdm(os.listdir(foreground_folder), desc="Loading foreground images")}

    print("正在加载背景图像...")
    background_images = {filename: load_image(os.path.join(background_folder, filename)) for filename in tqdm(os.listdir(background_folder), desc="Loading background images")}

    print("开始处理图像...")
    for background_filename, background in tqdm(background_images.items(), desc="Processing backgrounds"):
        if background is None:
            continue
        background_height, background_width = background.shape[:2]
        label_file = create_yolo_label(os.path.splitext(background_filename)[0] + ".txt", labels_folder)
        if label_file is None:
            continue
        composited_image = background.copy()
        mask_image = np.zeros_like(composited_image[:, :, 0], dtype=np.uint8)

        for _ in range(num_composites):
            foreground_filename = random.choice(list(foreground_images.keys()))
            foreground = foreground_images[foreground_filename]

            if foreground is None:
                continue

            foreground_height, foreground_width = foreground.shape[:2]
            foreground = resize_img(foreground, foreground_height, foreground_width, background_height, background_width, min_ratio, max_ratio)
            foreground = rotation(foreground)
            foreground_height, foreground_width = foreground.shape[:2]

            position = (random.randint(0, background_width - foreground_width),
                        random.randint(0, background_height - foreground_height))

            coords = find_opaque_coords(foreground)

            composited_image, mask_image = composite_image(coords, foreground, composited_image, position, mask_image)
            write_yolo_labels(position, label_file, foreground_height, foreground_width, background_height, background_width)

        # Save the composited image and mask
        os.makedirs(output_folder, exist_ok=True)
        composited_image_path = os.path.join(output_folder, f"{background_filename}")

        os.makedirs(mask_folder, exist_ok=True)
        mask_save_path = os.path.join(mask_folder, os.path.splitext(background_filename)[0] + ".png")

        cv2.imwrite(composited_image_path, composited_image)
        cv2.imwrite(mask_save_path, mask_image)
