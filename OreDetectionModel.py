import os
import cv2
import shutil
import random
from ultralytics import YOLO

# set splitting variables
ROOT_DIR = 'C:\\Users\\baid2\\Downloads\\Minecraft Ore Detection.v6i.yolov8' # Directory of dataset
train = "train"
val = "valid"
test = "test"
val_size = 0.1
test_size = 0.2

# training hyperparameters
epoch_count = 100
max_train_time = 3 # overrides epoch setting when training reaches max_train_time hours
patience = 10 # number of epochs to wait without any improvement to stop training early, helps with overfitting
batch_size = 8 # how many images are processed before parameters are updated
imgsz = 640 # resize images to target image size before training
optimizer = 'auto' # SGD(lr=0.01, momentum=0.9) with parameter groups 97 weight(decay=0.0), 104 weight(decay=0.0005), 103 bias(decay=0.0)
dropout = 0.5

# list of all images contained in training folder
image_files = os.listdir(os.path.join(ROOT_DIR, train + "\\images"))

# convert list of images to list of image names (remove image extension .jpg from name)
file_names = []
for img_name in image_files:
    file_names.append(img_name[:-4])

def TrainValTestSplit(val_size, test_size, image_files):
    valid_size = int(len(image_files) * val_size)
    test_size = int(len(image_files) * test_size)

    files_shuffled = file_names
    # shuffle image list
    random.shuffle(files_shuffled)
    print(files_shuffled)

    # if validation folder doesn't exist, create validation set from training set of size val_size
    if (not os.path.exists(os.path.join(ROOT_DIR, val))):
        new_image_path = os.path.join(ROOT_DIR, val + "\\images")
        new_label_path = os.path.join(ROOT_DIR, val + "\\labels")
        os.makedirs(new_image_path)
        os.makedirs(new_label_path)
        for file in files_shuffled[:valid_size]:
            shutil.move(os.path.join(ROOT_DIR, train + "\\images\\" + file + ".jpg"), new_image_path) # move training images to validation images folder
            shutil.move(os.path.join(ROOT_DIR, train + "\\labels\\" + file + ".txt"), new_label_path) # move training labels to validation labels folder

    # if test folder doesn't exist, create test set from training set of size test_size
    if (not os.path.exists(os.path.join(ROOT_DIR, test))):
        new_image_path = os.path.join(ROOT_DIR, test + "\\images")
        new_label_path = os.path.join(ROOT_DIR, test + "\\labels")
        os.makedirs(new_image_path)
        os.makedirs(new_label_path)
        for file in files_shuffled[valid_size:valid_size + test_size]:
            shutil.move(os.path.join(ROOT_DIR, train + "\\images\\" + file + ".jpg"), new_image_path) # move training images to test images folder
            shutil.move(os.path.join(ROOT_DIR, train + "\\labels\\" + file + ".txt"), new_label_path) # move training labels to test labels folder

def TrainModel():
    # Load a model
    #model = YOLO("yolov8x.yaml")

    # Load a model
    model = YOLO("OreDetectionModelv2.pt")

    # train the model
    results = model.train(data=os.path.join(ROOT_DIR, "data.yaml"),
                          epochs=epoch_count,
                          patience = patience,
                          time = max_train_time,
                          batch = batch_size,
                          imgsz = imgsz,
                          #optimizer = optimizer,
                          #dropout = dropout,
                          device = 0
                          )
    
    model.val() # evaluate performance on validation set

def DisplayTrainingData():
    # get random index for data
    rand_data = random.randrange(0, len(file_names))

    # get image of random index
    image = cv2.imread(os.path.join(ROOT_DIR, train + "\\images\\" + file_names[rand_data] + ".jpg"))

    # get image data path of random index
    image_data_path = os.path.join(ROOT_DIR, train + "\\labels\\" + file_names[rand_data] + ".txt")

    # read image data into arrays
    image_bbox = [] # 2d array of bounding boxes for each object
    image_cl = [] # classification values for each object
    with open(image_data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_cl.append(int(line[0])) # append classification value

            temp = []
            for bbox in (line.split()[1:]): # for each x_center, y_center, width, height
                temp.append(float(bbox)) # turn into a float and append to temp list
            image_bbox.append(temp) # append temp list as a bounding box


    # for each object, extract 
    for i in range(0, len(image_cl)):
        cl = image_cl[i] # get object classification
        bbox = image_bbox[i] # get image bbox
        top_left = (int((bbox[0] - (bbox[2]/2)) * 1920), int((bbox[1] - (bbox[3]/2)) * 1080)) # get the top left pixel of bounding box
        bot_right = (int(top_left[0] + (bbox[2] * 1920)), int(top_left[1] + (bbox[3] * 1080))) # get the bottom right pixel of bounding box
        color = (0, 255, 0) # bounding box color

        image = cv2.rectangle(image, top_left, bot_right, color, thickness=2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
def main():
    # If either validation set or test set or both doesn't exist, create it/them
    if(not os.path.exists(os.path.join(ROOT_DIR, val)) or not os.path.exists(os.path.join(ROOT_DIR, test))):
        TrainValTestSplit(val_size=val_size, test_size=test_size, image_files=image_files)

    DisplayTrainingData()
    TrainModel()

if __name__=="__main__": 
    main() 
