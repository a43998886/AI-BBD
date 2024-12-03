import os
import albumentations as A
import cv2

# Image folder path
image_folder = r''
# Label folder path
label_folder = r''
# Output image path
output_images_folder = r''
# Output label path
output_labels_folder = r''

transform = A.Compose([
    # A.HorizontalFlip(p=1.0),  # Horizontal flip
    # A.VerticalFlip(p=1.0),  # Vertical flip
    # A.RandomBrightnessContrast(p=1.0),  # Random brightness and contrast adjustment
    # A.RandomRotate90(p=1.0)  # Random 90-degree rotation
    # A.ColorJitter(p=1.0)  # Color jitter
    # A.RandomRain(p=1.0, blur_value=1, rain_type='drizzle')  # Random rain
    # A.RandomShadow(p=1.0)   # Random shadow
], bbox_params=A.BboxParams(format='yolo'))    # Set bounding box format


# Function to load image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to load label (YOLO format)
def load_label(label_path):
    bboxes_list = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            label = parts[0]
            coordinates = [float(coord) for coord in parts[1:]]
            bbox = coordinates + [label]
            bboxes_list.append(bbox)
    return bboxes_list

# Function to save label (YOLO format)
def save_label(label_path, bboxes):
    with open(label_path, 'w') as file:
        for bbox in bboxes:
            bbox = list(bbox)
            bbox_with_label_first = [bbox[-1]] + bbox[:-1]
            bbox_str = ' '.join(map(str, bbox_with_label_first))
            file.write(bbox_str + '\n')


if __name__ == '__main__':
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        bboxes = []
        if filename.endswith(".jpg") or filename.endswith(".png"):

            image_filename = os.path.splitext(filename)[0]
            image_path = os.path.join(image_folder, filename)

            label_filename = image_filename + ".txt"
            label_path = os.path.join(label_folder, label_filename)

            if os.path.exists(label_path):

                image = load_image(image_path)
                bboxes = load_label(label_path)

                transformed = transform(image=image, bboxes=bboxes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']

                transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_images_folder, filename), transformed_image)

                output_label_path = os.path.join(output_labels_folder, label_filename)
                save_label(output_label_path, transformed_bboxes)





