### Loading the Model Checkpoint

To perform face identification using the pre-trained InceptionV2 model checkpoint, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TrickyWhiteCat/Face-Identification.git
   cd Face-Identification

2. **Download the Checkpoint: (Skip this)**
   
   Download the InceptionV2 model checkpoint (InceptionV2.pth) from this [link](https://husteduvn-my.sharepoint.com/:u:/g/personal/tuan_nm214940_sis_hust_edu_vn/EQO3kBG3TRtJssQwbI3YL6MBYnt9uJKRVyq_25LQBvK7iA?e=66tA7I) and place it in the verification/triplet/inceptionv2/result/ directory.
   However, I already have it on Github, so you can skip this step.

4. **Load the Checkpoint:**
```
import torch
from verification.triplet.inceptionv2.model import InceptionV2

# Create an instance of the InceptionV2 model
model = InceptionV2(num_classes=your_num_classes)  # Replace with the actual number of classes

# Specify the path to the checkpoint file
checkpoint_path = 'verification/triplet/inceptionv2/result/InceptionV2.pth'

# Load the model checkpoint
loaded_model = load_model_checkpoint(model, checkpoint_path)
```

Ensure to replace your_num_classes with the actual number of classes.

## Training the Model

To train the InceptionV2 model for face verification, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TrickyWhiteCat/Face-Identification.git
   cd Face-Identification
2. **Set Up the Dataset:**

   Prepare your dataset and set the root folder path in the verification/triplet/inceptionv2/train/train.py file.

   Adjust the transformations in train_transform_color and valid_transform based on your dataset requirements.

3. **Run Training Script:**

   Navigate to the verification/triplet/inceptionv2/train/ directory.

   Execute the training script train.py using the following command:
   ```bash
   python train.py

   You can customize the training parameters within the script, such as batch size, the number of workers, and others.

4. **Monitor Training Progress:**

   During training, you can monitor the training progress and metrics on Weights & Biases (of course your WanDB :), replace in train.py code).
   The trained model checkpoint will be saved as "InceptionV2.pth" in the current directory.

5. **Resume Training (Optional):**

   If you want to resume training from a specific checkpoint, set the resume_checkpoint variable to the desired checkpoint name.

6. **Early Stopping (Optional):**

   Enable early stopping by setting the early_stop_patience parameter to a positive value. Training will stop if the validation loss does not improve for the specified number of epochs.

7. **Customize Training (Optional):**

   Feel free to customize the training script further based on your specific requirements.

