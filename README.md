# Face Detection

 
- Wandb Training FaceDetection Model: https://wandb.ai/doanngoccuong_nh/Yolov5_FaceDetection?workspace=user-doanngoccuong

- Run infer.ipynb trong colab để test detection nhanh, cropped face

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

# My API: c8767797aae76cbcd389ff29929ace1ac3021161
```