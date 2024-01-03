# Face Detection

- Dataset:  WIDERFACE Dataset   http://shuoyang1213.me/WIDERFACE/index.html
WIDER FACE dataset is used for face detection. It includes 32,203 images with 393,703 labeled faces. These faces vary in scale, pose, and degree of concealment. The data is organized into 61 event types, divided into 40% for training, 10% for testing, and 50% for testing.
- Model: Yolov5s with Train: 12880 images(40%), Val: 3226 images(10%), Test: (50%)
    - Training (use cfg, weights null)
        ```python
        # Train từ scratch, để img 640, trước để 416
        !python train.py --img 640 --batch 16 --epochs 10 \
        --data /content/drive/MyDrive/colab/custom_dataset/custom_dataset.yaml \
        --cfg /content/drive/MyDrive/colab/yolov5/models/yolov5s.yaml \
        --weights '' \
        --name facedet_widerface_cfgyolov5_colab --cache \
        --project Yolov5_FaceDetection

        ```

    - Processing Training is logged at Wandb, see: https://wandb.ai/doanngoccuong_nh/Yolov5_FaceDetection?workspace=user-doanngoccuong

```python
!git clone https://github.com/ultralytics/yolov5.git
!cd yolov5 && pwd
%cd yolov5
!pip install -r requirements.txt
```

```python
# Load best_training model
!pip install wandb
import wandb
run = wandb.init()
artifact = run.use_artifact('doanngoccuong_nh/Yolov5_FaceDetection/run_y9gaafqy_model:v0', type='model')
artifact_dir = artifact.download()
print(artifact_dir)

# My API: c8767797aae76cbcd389ff29929ace1ac3021161
# link weights_best.pt: /content/yolov5/artifacts/run_y9gaafqy_model:v0/best.pt
```

```python
# import file for test
%mkdir /content/real_test
%cd /content/real_test
from google.colab import files
files.upload()
```

```python
# Infer real_test, optinonal:  --img 640 --save-txt --save-conf (resize, save txt, save confident)
%cd /content/yolov5
!python detect.py --source /content/real_test --weights /content/yolov5/artifacts/run_y9gaafqy_model:v0/best.pt  --save-txt --save-conf

```

```python
### Crop Detected Face: use face_img = image[y1:y2, x1:x2] = image[y_min:y_max, x_min:x_max]
import os
import cv2

image_folder = '/content/real_test'
labels_folder = '/content/yolov5/runs/detect/exp2/labels'
output_folder = '/content/cropped_faces'
os.makedirs(output_folder, exist_ok=True)

# Lặp qua each image in image_folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Read image from full path
        path = os.path.join(image_folder, filename)
        image = cv2.imread(path)

        # Tên file txt tương ứng (cùng tên với ảnh)
        txt_file = filename.replace('.jpg', '.txt').replace('.png', '.txt')

        # Read file txt to get tọa độ bounding boxes
        with open(os.path.join(labels_folder, txt_file), 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                _, x_center, y_center, width, height = map(float, line.split()[:5])
                # Tình toán lại tọa độ bounding box theo kích thước ảnh gốc (unnormalize)
                x_center, width = x_center * image.shape[1], width * image.shape[1]
                y_center, height = y_center * image.shape[0], height * image.shape[0]
                # chuyển tọa độ bounding box sang x1, y1, x2, y2 (x1, y1 là góc trên bên trái, x2, y2 là góc dưới bên phải) or x_min, y_min, x_max, y_max
                x_min, y_min, x_max, y_max = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)

                # Crop và lưu ảnh khuôn mặt
                face_img = image[y_min:y_max, x_min:x_max] # image[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(output_folder, f'{filename[:-4]}_face_{i}.jpg'), face_img)
        print(f"Cropping successful {filename}")
print("Cropping hoàn tất.")



```