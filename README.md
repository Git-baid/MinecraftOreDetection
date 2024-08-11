# Minecraft Ore Detection With Machine Learning

**This is a class project for CSCE A415 Machine Learning at the University of Alaska Anchorage**

<img src="https://github.com/Git-baid/MinecraftOreDetection/blob/main/Minecraft%20Ore%20Detection%20(1).gif" width = 640>
*Classify and localize Minecraft ores in real-time using a hand-made, hand-annotated, curated dataset of over 1,200 images*
## Introduction
The goal of this project is to classify and localize minecraft ores in real-time from a stream of screenshots on a user’s computer. I started the project using the dataset in a COCO format, but later discovered the YOLO object detection framework which fit exactly what we were trying to accomplish much better and with less friction. The model is based on YOLOv8, a popular choice and evolution for object detection framework. The model is trained using a custom dataset of 1,211 hand annotated images of minecraft ores using [Roboflow.com](www.roboflow.com)’s custom dataset and annotation tools. 

## YOLOv8 Overview
YOLOv8 is an evolution of the YOLO (You Only Look Once) family of object detection and segmentation models. YOLOv8 is designed to be fast, accurate, flexible, and easy to use.
- Speed: YOLOv8 is designed for real-time object detection, making it suitable for the goal of real-time minecraft ore detection.
- Accuracy: YOLOv8 is highly accurate for object detection tasks while remaining efficient and fast
- Flexibility: YOLOv8 can be customized for different datasets and tasks, making it useful for the custom dataset and goal.
For these three reasons, YOLO object detection was chosen to be the best choice for the base detection framework.

## Dataset

![image](https://github.com/user-attachments/assets/69605cc2-d817-4fd5-82f2-da256ed86123)

- The dataset consists of 1,211 hand annotated images of 5 different minecraft ores (Deepslate Diamond, Deepslate Redstone, Deepslate Gold, Deepslate Iron, Iron Ore) with about 250 images per class. Minecraft currently has 19 different ore types in but I chose to focus on only the most relevant ores for the scope of this project.
- The dataset folder is then split into a training, validation, and test set before training begins if it detects those folders don’t exist. The size of these sets are variables preset to 20% for validation and 10% for testing.
- Taking the images and annotating the custom dataset took an extensive amount of the time spent on the project (even with some help), with 1,211 images and each image often having multiple instances of multiple different classes to annotate. 250 images per class in a dataset is a relatively low number for a robust model. I believe the performance of the model is largely bottlenecked by the size and quality of the dataset. With more time and a larger team, I think the model can be vastly improved by increasing the number of classes, total images per class, and by using more variety in the environment.

## Training
Using VSCode IDE with Python 3.12
- On top of the extra large (slower but more accurate) size YOLOv8 model as a base, I trained the final version of our model for ~80 epochs over the span of 8 and a half hours. Around 10 minutes per epoch.
- The training uses Binary Cross-entropy for its classification branch loss function and CIoU for its bounding boxes. The Binary Cross-entropy loss function helps give the model a confidence score in its classification which is used to rule out low confidence classifications in real time. CIoU (Complete Intersection over Union) scores the area of overlap of the predicted bounding box and true bounding box. It also takes into account the distance of their centers and aspect-ratio.
- Stochastic Gradient Descent was used as an optimizer during training with a learning rate of 0.01. I think that exploring more optimizer options with different learning rates can potentially lead to some improvement to the model, but due to the long training times, I found this to be a good solution without spending too much time exploring every option.
- Images were resized to 640x640 before training. Image resolutions in the dataset were not consistent so this is an important step not only to create a consistent training set but to scale down large images in the dataset for faster training. In hindsight, a higher training resolution might have been beneficial because the model is not great at detecting ores at a distance, or ores at extreme angles.

## Evaluation Metrics
![image](https://github.com/user-attachments/assets/564623a9-040a-4bd4-a962-617aeecad39c)
![image](https://github.com/user-attachments/assets/34e00c7c-8a08-4f4b-ac2a-62451e3dcab6)

As you can see in the first three graphs, there is a nice quick decline in the loss values during training and a slow taper out as it converges. It looks like it is not overfitting with a nice consistent curvature towards convergence. And a similar upward trend of our mean Average Precision.

## Areas for Improvement

**Dataset**: I think the bottleneck of the project’s performance is the dataset, with more images per class with more classes to span all the ores in the game, we can significantly increase accuracy and performance of our model. Because 250 images per class is relatively low for a robust dataset. Also including dataset images with different lighting conditions would improve versatility. 

**Low confidence scores**: This can be improved by expanding the dataset and potentially training using higher resolutions to help very slim perspective detections and detections further away. 

**Framerate**: This detection runs real-time in a separate window at about 10 frames per second. This framerate is likely bottlenecked in the code that displays the resulting detections in a separate window because for each frame displayed, it is taking a screenshot and saving it to disk and loading the screenshot into the Pillow image library. It is not a fault of the model itself, but that is definitely a possible area of improvement. It is kind of a hacky way to go about displaying the resulting image but I wanted most of the focus to be on the model and dataset themselves.

Overall, I'm happy with the result and believe that it serves as a fun strong foundational project for machine learning that can possibly be expanded on and improved upon with more time and resources in the future.


Dataset download: https://universe.roboflow.com/ore-detection/minecraft-ore-detection-20pzg
