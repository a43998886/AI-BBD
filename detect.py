import os
import cv2
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('models/lws.pt')


def slice(image_path, image_size=(640, 640), over_lap_rate=0.1, visualization=False):  # image_size:(H，W）

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    tensor_list = []

    cut_amount_row = (width // image_size[1]) + 1
    cut_amount_column = (height // image_size[0]) + 1

    for i in range(cut_amount_column):
        for j in range(cut_amount_row):
            Topleft_x = int(j * (1 - over_lap_rate) * image_size[1])
            Topleft_y = int(i * (1 - over_lap_rate) * image_size[0])
            Bottomright_x = Topleft_x + image_size[1]
            Bottomright_y = Topleft_y + image_size[0]
            cropped_img = img[Topleft_y:Bottomright_y, Topleft_x:Bottomright_x]

            if cropped_img.shape[:2] != (image_size[0], image_size[1]):
                bottom_pad = max(0, image_size[0] - cropped_img.shape[0])
                right_pad = max(0, image_size[1] - cropped_img.shape[1])

                cropped_img = cv2.copyMakeBorder(cropped_img, 0, bottom_pad, 0, right_pad,
                                                 cv2.BORDER_CONSTANT, value=0)

            img_tensor = ToTensor()(cropped_img)
            tensor_list.append(img_tensor)

    tensor_data = torch.stack(tensor_list)
    # -------------------------------------------------------------------------------------------------------
    if visualization:
        fig, axs = plt.subplots(cut_amount_column, cut_amount_row, figsize=(20, 20))

        for i in range(cut_amount_column):
            for j in range(cut_amount_row):
                idx = i * cut_amount_row + j
                if idx < tensor_data.shape[0]:
                    axs[i, j].imshow(tensor_data[idx].permute(1, 2, 0))
                    axs[i, j].axis('off')
        plt.show()
    # -------------------------------------------------------------------------------------------------------
    return img, tensor_data, cut_amount_row


def getboxes(results, cut_amount_row, over_lap_rate, image_size=(640, 640)):
    boxes_list = []
    for k, result in enumerate(results):

        if len(result.boxes) != 0:
            if (k + 1) % cut_amount_row == 0:
                cut_x = ((k + 1) // cut_amount_row) - 1
                cut_y = cut_amount_row - 1
            elif (k + 1) < cut_amount_row:
                cut_x = 0
                cut_y = k
            else:
                cut_x = ((k + 1) // cut_amount_row)
                cut_y = ((k + 1) % cut_amount_row) - 1

            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                box_left_x, box_left_y, box_right_x, box_right_y = box.tolist()
                bigpic_box_left_x = int(box_left_x + cut_y * (1 - over_lap_rate) * image_size[1])  # 左上x坐标
                bigpic_box_left_y = int(box_left_y + cut_x * (1 - over_lap_rate) * image_size[0])  # 左上y坐标
                bigpic_box_right_x = int(box_right_x + cut_y * (1 - over_lap_rate) * image_size[1])  # 右下x坐标
                bigpic_box_right_y = int(box_right_y + cut_x * (1 - over_lap_rate) * image_size[0])  # 右下y坐标

                boxes_list.append([bigpic_box_left_x, bigpic_box_left_y, bigpic_box_right_x,
                                   bigpic_box_right_y, round(float(conf), 2)])

    return boxes_list


def draw_boxes_and_save(img, iltered_boxes, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in iltered_boxes:
        cv2.rectangle(img, (i[0] - 5, i[1] - 5), (i[2] + 5, i[3] + 5), (0, 0, 255), 4)
        confidence_text = f"{i[4]:.2f}"
        cv2.putText(img, confidence_text, (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.line(img, (0, 0), (i[0], i[1]), (0, 255, 0), 2)
    cv2.imwrite(save_path, img)


def find_images(root_folder):
    image_files = []
    for folder_name, _, file_names in os.walk(root_folder):
        for file_name in file_names:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(folder_name, file_name)
                image_files.append(image_path)
    return image_files


if __name__ == '__main__':
    root_folder_path = r" "
    output_path = r"results"
    right_detection_dir = os.path.join(output_path, 'detection')
    no_detection_dir = os.path.join(output_path, 'no_detection')
    img_paths_list = find_images(root_folder_path)
    over_lap_rate = 0.1
    img_size = (640, 640)
    iou_threshold = 0
    test_number = 20
    pic_num = len(img_paths_list)

    for i, img_path in enumerate(img_paths_list):

        img, yolo_data, cut_amount_row = slice(img_path)
        results = model.predict(yolo_data, imgsz=img_size, conf=0.5, save=False)
        boxes_list = getboxes(results, cut_amount_row, over_lap_rate)

        if len(boxes_list) != 0:
            draw_boxes_and_save(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), boxes_list,
                                os.path.join(right_detection_dir, os.path.relpath(img_path, root_folder_path)))

        else:
            if not os.path.exists(no_detection_dir):
                os.makedirs(no_detection_dir)
            cv2.imwrite(os.path.join(no_detection_dir, os.path.basename(img_path)),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if i == test_number - 1:
            pic_num = test_number
            break
