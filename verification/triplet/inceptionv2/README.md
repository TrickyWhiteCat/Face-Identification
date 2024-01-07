### Loading the Model Checkpoint

To perform face identification using the pre-trained InceptionV2 model checkpoint, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TrickyWhiteCat/Face-Identification.git
   cd Face-Identification

2. **Download the Checkpoint: (Skip this)**
Download the InceptionV2 model checkpoint (InceptionV2.pth) from this [link](https://husteduvn-my.sharepoint.com/:u:/g/personal/tuan_nm214940_sis_hust_edu_vn/EQO3kBG3TRtJssQwbI3YL6MBYnt9uJKRVyq_25LQBvK7iA?e=66tA7I) and place it in the verification/triplet/inceptionv2/result/ directory.
However, I already have it on Github, so you can skip this step.

3. **Load the Checkpoint:**
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
**Ensure to replace your_num_classes with the actual number of classes.**
