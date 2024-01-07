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

    - Processing Training is logged at Wandb, see: https://wandb.ai/doanngoccuong_nh/FaceDetection_Yolov5?workspace=user-doanngoccuong

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
run.finish()    # Finish logging to W&B (WandB) immediately after load model succesful. 
 
<<<<<<< HEAD
# My API: c8767797aae76cbcd389ff29929ace1ac3021161

```

```python
# import file for test
%mkdir /content/real_test
%cd /content/real_test
from google.colab import files
files.upload()
```

```python
# Infer real_test, 
# optional:  --img 640 --save-txt --save-conf (resize, save txt, save confident)
# optional: --save-csv: predictions.csv (face, conf). --save-crop 
# OR Crop Detected Face use labels file txt: face_img = image[y1:y2, x1:x2] = image[y_min:y_max, x_min:x_max]

%cd /content/yolov5
!python detect.py --source /content/real_test --weights /content/yolov5/artifacts/run_y9gaafqy_model:v0/best.pt  --save-txt --save-conf

!python detect.py --source /content/real_test \
                  --weights /content/yolov5/artifacts/run_y9gaafqy_model:v0/best.pt \
                  --save-txt --save-conf --save-crop --save-csv
```

=======
MTCNN và FaceNet là 2 mạng rất nổi tiếng trong việc xử lý bài toán Face Recognition nói chung. Và việc kết hợp giữa chúng, khi đầu vào là ảnh/video với rất nhiều người và trong hoàn cảnh thực tế, sẽ đưa ra được kết quả khá tốt. Khi đó, MTCNN sẽ đóng vai trò là Face Detection/Alignment, giúp cắt các khuôn mặt ra khỏi khung hình dưới dạng các tọa độ bounding boxes và chỉnh sửa / resize về đúng shape đầu vào của mạng FaceNet. Còn FaceNet sẽ đóng vai trò là mạng Feature Extractor + Classifier cho từng bounding boxes, đưa ra embedding và tiền hành phân biệt và nhận dạng các khuôn mặt. Ở bài tiếp theo, chúng ta sẽ tìm hiểu về cách inference lại cả 2 mạng trên và tạo ra một mạng hoàn chỉnh, giúp nhận diện Realtime danh tính khuôn mặt nhé!
>>>>>>> 11e766654a286d77ecaddc2f26f637ab1c7979ed
