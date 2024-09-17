
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
from util.filter import gaussian_fuzzy_filter



def crop_resize_and_mask(image_path,bbox,output_path):
    img = Image.open(image_path).convert('L')  # 转换为灰度图像

    width, height = img.size  # 获取图像的宽度和高度

    # 获取 bbox 信息
    x_center, y_center, bbox_width, bbox_height = bbox

    # 计算中心点及框的宽度、高度在原图像中的像素坐标
    x_center_pixel = int(x_center * width)
    y_center_pixel = int(y_center * height)
    bbox_width_pixel = int(bbox_width * width)
    bbox_height_pixel = int(bbox_height * height)

    # 确定较长边
    longer_side = max(bbox_width_pixel, bbox_height_pixel)
    # 正方形边长是较长边的1.2倍
    square_side = int(longer_side * 1.2)

    # 计算正方形框的左上角和右下角的坐标
    left = int(x_center_pixel - square_side / 2)
    top = int(y_center_pixel - square_side / 2)
    right = int(x_center_pixel + square_side / 2)
    bottom = int(y_center_pixel + square_side / 2)

    # 边界检查，确保正方形框不会超出图像范围
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    # 进行裁剪
    img_cropped = img.crop((left, top, right, bottom))

    # 目标尺寸 (192x128)
    target_width = 224
    target_height = 224  # 这里修改为正方形尺寸

    # 将裁剪的图像等比例缩放到目标尺寸
    img_np = np.array(img_cropped)

    # 进行降噪处理
    img_filtered = gaussian_fuzzy_filter(img_np)

    img_resized = cv2.resize(img_filtered, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # 转换为 PIL 格式并保存
    img_output = Image.fromarray(img_resized)

    # 保存处理后的图像
    base_ = os.path.basename(image_path)
    fo = os.path.splitext(base_)[0]
    img_output.save(os.path.join(output_path, fo + '_cropped.jpg'))

    return


def process_images_and_labels(image_folder, label_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图像文件夹中的每个图像文件
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 获取图像文件的基础名称（不带扩展名）
            base_name = os.path.splitext(filename)[0]

            # 构建对应的 labels 文件路径
            label_file_path = os.path.join(label_folder, base_name + '.txt')
            print(label_file_path)
            # 如果 labels 文件存在，读取 bbox 数据
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as f:
                    bbox_data = f.read().strip().split(' ')

                # 将数据转换为浮点数，并取最后四个数作为 bbox
                bbox = list(map(float, bbox_data[-4:]))

                # 调用处理函数
                image_path = os.path.join(image_folder, filename)
                crop_resize_and_mask(image_path, bbox, output_folder)
            else:
                pass
                # print(f"Label file not found for {filename}")

# # 使用示例
image_folder = 'E:/Dataset/Contest4/Ultrasound_breast_data/test_A/feature-test/A/images'  # images 文件夹路径
label_folder = 'E:/Dataset/Contest4/Ultrasound_breast_data/test_A/feature-test/A/labels' # labels 文件夹路径
output_folder = 'E:/Dataset/Contest4/Ultrasound_breast_data/test_A/feature-test/A/cropped_images'  # 输出文件夹路径
process_images_and_labels(image_folder, label_folder, output_folder)




label_map = {
    "boundary": 0,
    "calcification": 1,
    "direction": 2,
    "shape": 3
}
def process_features_labels(parent_path):
    img_path = os.path.join(parent_path,'images')
    boundary_labels = os.path.join(parent_path + '/boundary_labels')
    calcification_labels = os.path.join(parent_path + '/calcification_labels')
    direction_labels = os.path.join(parent_path + '/direction_labels')
    shape_labels = os.path.join(parent_path + '/shape_labels')
    output_path = os.path.join(parent_path + '/labels')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for filename in os.listdir(img_path):
        flag = 0
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 获取图像文件的基础名称（不带扩展名）
            base_name = os.path.splitext(filename)[0]

            # 1. boundary_label
            boundary_labels_path = os.path.join(boundary_labels,base_name + '.txt').replace('\\','/')
            if os.path.exists(boundary_labels_path):
                with open(boundary_labels_path, 'r') as f:
                    b_content = f.read().strip()
                    if not b_content:
                        boundary_label = 999
                        pass
                    else:

                        flag = 1
                        b_data = b_content.split(' ')
                        boundary_label = int(b_data[0])
                        bbox = list(map(float, b_data[-4:]))

            # 2. calcification_label
            calcification_labels_path = os.path.join(calcification_labels, base_name + '.txt')
            if os.path.exists(calcification_labels_path):
                with open(calcification_labels_path, 'r') as f:
                    c_content = f.read().strip()
                    if not c_content:
                        calcification_label = 999
                        pass
                    else:
                        c_data = c_content.split(' ')
                        calcification_label = int(c_data[0])
                        if flag == 0:
                            flag = 1
                            bbox = list(map(float, c_data[-4:]))

            # 3. direction_label
            direction_labels_path = os.path.join(direction_labels, base_name + '.txt')
            if os.path.exists(direction_labels_path ):
                with open(direction_labels_path, 'r') as f:
                    d_content = f.read().strip()
                    if not d_content:
                        direction_label = 999
                        pass
                    else:
                        d_data = d_content.split(' ')
                        direction_label = int(d_data[0])
                        if flag == 0:
                            flag = 1
                            bbox = list(map(float, d_data[-4:]))

            # 4. shape_label
            shape_labels_path = os.path.join(shape_labels, base_name + '.txt')
            if os.path.exists(shape_labels_path ):
                with open(shape_labels_path, 'r') as f:
                    s_content = f.read().strip()
                    if not s_content:
                        shape_label = 999
                        pass
                    else:
                        s_data = s_content.split(' ')
                        shape_label = int(s_data[0])
                        if flag == 0:
                            flag = 1
                            bbox = list(map(float, s_data[-4:]))

            bbox.insert(0,shape_label)
            bbox.insert(0,direction_label)
            bbox.insert(0,calcification_label)
            bbox.insert(0,boundary_label)

            written_path = os.path.join(output_path, base_name + '.txt').replace('\\','/')
            with open(written_path, 'w') as file:
                file.write(' '.join(map(str, bbox)))


# process_features_labels(parent_path='E:/Dataset/Contest4/Ultrasound_breast_data/feature_train/train')
# process_features_labels(parent_path='E:/Dataset/Contest4/Ultrasound_breast_data/test_A/feature-test/A')

