# utils
import os
import shutil
import json
import glob
import urllib
import tarfile
import math
import random
import numpy as np
import subprocess

# images
import cv2
import albumentations
import albumentations.pytorch

# Torch
import torch
import timm
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim


################################ Variables ################################
classes = ["A", "F", "TA", "PT", "DC", "LC", "MC", "PC"]
zooms = ["40", "100", "200", "400"]


################################ Classes ################################
class My_data(Dataset):
    def __init__(self, data, transforms=None, setnp=False):
        self.image_list = data
        self.data_len = len(self.image_list)
        self.transforms = transforms
        self.eicls = ["A", "F", "TA", "PT", "DC", "LC", "MC", "PC"]

    def __getitem__(self, index):       
        current_image_path = self.image_list[index]
        im_as_im = cv2.imread(current_image_path)
        if im_as_im is None:
            raise ValueError(f"Image not found or is corrupted: {current_image_path}")
        im_as_im = cv2.cvtColor(im_as_im, cv2.COLOR_BGR2RGB)

        # Perform label encoding for multi-label classification
        parts = current_image_path.split('_')[-1].split('-')
        if parts[2]=="13412":
            labels =[0,0,0,0,1,1,0,0]
        else:
            labels = [int(label == parts[0]) for label in self.eicls]
        labels = torch.tensor(labels)

        if self.transforms is not None:
            augmented = self.transforms(image=im_as_im)
            im_as_im = augmented['image']
            #print(type(im_as_im))

        return (im_as_im, labels)
    
    def __getimages__(self):
        images = []
        for i in range(0, len(self.image_list)):
            images.append(self.__getitem__(i)[0])
        return torch.stack(images, dim=0)

    def __getlabels__(self):
        labels = []
        for i in range(0, len(self.image_list)):
            labels.append(self.__getitem__(i)[1])
        return torch.stack(labels, dim=0)                
    
    def __getname__(self, index):
        return self.image_list[index].split(os.sep)[-1]
    
    def __getzoom__(self, index):
        return self.image_list[index].split("_")[-1].split("-")[3]
    
    def __getclass__(self, index):
        return self.image_list[index].split("_")[-1].split("-")[0]
    
    def __getpath__(self, index):
        return os.sep.join(self.image_list[index].split(os.sep)[-3:])

    def __len__(self):
        return self.data_len
    

class FocalLoss(nn.Module):
    def __init__(self, device, gamma=2.0, class_weights=torch.tensor([2.2232, 0.9754, 1.7378, 2.1744, 0.2868, 1.5779, 1.2469, 1.7619])):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.class_weights = self.class_weights.to(device)

    def forward(self, logits, labels):
        # probs = torch.sigmoid(logits)
        # ce_loss = nn.BCELoss()(probs, labels)
        # weight = (1 - probs).pow(self.gamma)
        # loss = ce_loss  # Initialize loss with cross-entropy loss

        # if self.class_weights is not None:
        #     weight = weight * self.class_weights
        #     loss = loss * weight
        # return loss

        # Calculate BCE loss without reduction
        bce_loss = nn.BCEWithLogitsLoss(weight=self.class_weights, reduction='none')(logits, labels)
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate the focal loss scaling factor
        focal_scaling = (1 - probs).pow(self.gamma)
        
        # Apply focal loss scaling factor to BCE loss
        focal_loss = focal_scaling * bce_loss

        # Aggregate the losses
        return focal_loss.mean()

    
    
class CustomTransforms():
    def __init__(self):
        self.transform = {
            'train': albumentations.Compose([
                albumentations.Resize(256, 256),
                albumentations.OneOf([
                                    albumentations.HorizontalFlip(),
                                    albumentations.Rotate(limit=45),
                                    albumentations.VerticalFlip(),
                                    albumentations.GaussianBlur(),
                                    albumentations.NoOp()
                ], p=1),
                albumentations.Normalize(mean=(0, 0, 0),
                                std=(255, 255, 255), max_pixel_value=1.0, p=1),
                albumentations.pytorch.transforms.ToTensorV2()
            ]),

            'valid': albumentations.Compose([
                albumentations.Resize(256, 256),
                albumentations.Normalize(mean=(0, 0, 0),
                                std=(255, 255, 255), max_pixel_value=1.0, p=1),
                albumentations.pytorch.transforms.ToTensorV2()
            ]),

            'test': albumentations.Compose([
                albumentations.Resize(256, 256),
                albumentations.Normalize(mean=(0, 0, 0),
                                std=(255, 255, 255), max_pixel_value=1.0, p=1),
                albumentations.pytorch.transforms.ToTensorV2()
            ]),

            'resize': albumentations.Compose([
                albumentations.Resize(256, 256),
            ]),

            "resize_tensor": albumentations.Compose([
                albumentations.Resize(256, 256),
                albumentations.pytorch.transforms.ToTensorV2()
            ]),
        }

    def get_transform(self, transformer):
        try:
            return self.transform[transformer]
        except KeyError:
            print("Transformer does not exist")      
    

################################ Functions ################################
#                                  CUDA                                   #
def get_gpu_memory():
    try:
        # Run the nvidia-smi command to get memory usage
        smi_output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True)
        gpu_memory = [int(x) for x in smi_output.decode('utf-8').strip().split('\n')]
        return gpu_memory
    except Exception as e:
        print(f"Error querying GPU memory: {e}")
        return []

def select_gpu():
    if torch.cuda.is_available():
        gpu_memory = get_gpu_memory()
        if gpu_memory:
            # Select the GPU with the most free memory
            gpu_id = gpu_memory.index(max(gpu_memory))
            print(f"Selecting GPU {gpu_id} with {gpu_memory[gpu_id]}MB free memory")
            return gpu_id
        else:
            print("Unable to get GPU memory. Using default GPU.")
            return 0
    else:
        print("CUDA not available. Using CPU.")
        return "cpu"

#                                  Data                                   #
def get_data(path):
    # Path is where the train test split data lives
    # Check whether the dataset exists on the system, if not, download and unzip the dataset
    if not os.path.isdir(os.path.join(os.getcwd(), "BreaKHis_v1")):
        urllib.request("http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz")
        with tarfile.open("BreaKHis_v1.tar.gz") as tar:
            tar.extractall("BreaKHis_v1")
    
    # Check the existance of the train.txt and test.txt files, if so, read them from the files
    if os.path.isfile(os.path.join(os.getcwd(), "txt", "train.txt")) and os.path.isfile(os.path.join(os.getcwd(), "txt", "test.txt")):
        train_dict = get_dict(os.path.join(os.getcwd(), "txt", "train.txt"))
        test_dict = get_dict(os.path.join(os.getcwd(), "txt", "test.txt"))
        correct = check_corrupted_imgs(train_dict, test_dict)
    # if not, create the train test split
    else:
        train_dict, test_dict = create_train_test_split(path)

    return train_dict, test_dict


def create_train_test_split(data_path):
    all_files = get_files(data_path)
   
    # Create dictionary that filters images based on class and zoom level
    data_dict = {c: {z: [path for path in all_files if path.split("_")[-1].split("-")[0] == c and path.split("_")[-1].split("-")[3] == z] for z in zooms} for c in classes}

    # Initialize empty dicts
    train_dict = {c: {z: [] for z in zooms} for c in classes}
    test_dict = {c: {z: [] for z in zooms} for c in classes}

    # Make train/test split
    train_test_split = 0.9

    for c in data_dict.keys():
        for z, v in data_dict[c].items():
            split = math.ceil(train_test_split * len(v))

            # Randomly sample from file paths for given class/zoom level
            train_images = random.sample(data_dict[c][z], split)

            # Take as test data all files that are not in the train data for a given class/zoom level
            test_images = np.setdiff1d(data_dict[c][z], train_images)

            # Check if error is made
            if len(train_images) + len(test_images) != len(v):
                print("Error in train/test split at {}-{}".format(c, z))

            # Store train and test data in dictionaries
            train_dict[c][z] = list(train_images)
            test_dict[c][z] = list(test_images)

    for c in classes:
        for z in zooms:
            print("Class {}, Zoom {}".format(c, z))
            print("Total: {}, Train: {}, Test: {}".format(len(data_dict[c][z]), len(train_dict[c][z]), len(test_dict[c][z])))

    with open(os.path.join("txt", "train.txt"), "w") as f:
        json.dump(train_dict, f)

    # Create test text file
    with open(os.path.join("txt", "test.txt"), "w") as f:
        json.dumps(test_dict, f)

    return train_dict, test_dict


# Takes all images from a train/test dict and copies them to new destination for easy access
def copy_images(image_dict, type):

    # Get current working directory
    current_dir = os.getcwd()

    # Check whether the new file structure is already initialised
    if not os.path.isdir(os.path.join(current_dir, "dataset", "train", "TA", "400")) or not os.path.isdir(os.path.join(current_dir, "dataset", "test", "TA", "400")):
        for c in classes:
            for z in zooms:
                os.makedirs(os.path.join(os.getcwd(), "dataset", "test", c, z))
                os.makedirs(os.path.join(os.getcwd(), "dataset", "train", c, z))

    for c in classes:
        for z in zooms:

            # Destination path
            dst = os.path.join(current_dir, "dataset", type, c, z)

            # Remove all images currently in destination path
            for file in os.listdir(dst):
                os.remove(os.path.join(dst, file))

            # Copy all images from BreaKHis to dataset directory
            for image in image_dict[c][z]:
                src = os.path.join(image)
                dst = os.path.join(current_dir, "dataset", type, c, z)

                shutil.copy2(src, dst)

def make_dirs(name):
    if not os.path.isdir(os.path.join(os.getcwd(), "dataset", name, "TA", "400")):
        for c in classes:
            for z in zooms:
                os.makedirs(os.path.join(os.getcwd(), "dataset", name, c, z))


def get_dict(filepath):
    with open(os.path.join(os.getcwd(), "txt", filepath)) as f:
        print(f"Opening {filepath}.txt")
        return json.load(f)
    

def get_files(filepath):
    return glob.glob(filepath)
    

def check_corrupted_imgs(train_dict, test_dict):
    """This code checks the image paths for a correct split. If the split seems corrupted, the function calls copy_images which ensures a proper split.
    Input: the train dict retreived from the text file."""
    correct = True
    copied_files = glob.glob("./dataset/train/A/40/*.png")
    copied_files = [path.split("_")[-1] for path in copied_files]

    if len(train_dict["A"]["40"]) != len(copied_files):
        print("Not as many A-40 train files in dict as in dataset directory")
        print("Therefore current copy not correct")
        correct = False
    else:
        for path in train_dict["A"]["40"]:
            if path.split("_")[-1] not in copied_files:
                print("File {} found in train dict, but not in copied files.".format(path.split("_")[-1]))
                print("Therefore current copy not correct")
                correct = False

    if not correct:
        print("Current copy does not satisfy.")
        print("Copying images from BreaKHis to dataset/** directory according to train_dict and test_dict.")

        # Copy images to correct directory for easy access
        copy_images(train_dict, "train")
        copy_images(test_dict, "test")

        correct = check_corrupted_imgs(train_dict, test_dict)

    return correct
    

def save_image(tensor, path):
    # Transform tensor to PIL Image
    image = transforms.ToPILImage()(tensor)

    # Save image
    image.save(path)


def perturb_image(perturbation, img):
    y_pos, x_pos, *rgb = perturbation
    
    img[y_pos, x_pos] = rgb

    return img

#                                  Model                                  #
def get_class_weigths(image_dict: dict):
    # Number of samples in each class
    # Assumption is made that in each fold, these are the same (since we do stratified split)
    class_samples = [sum([len(z) for z in c.values()]) for c in image_dict.values()]

    # Hardcoded from paper
    # class_samples = [367, 803, 456, 370, 2763, 492, 629, 449]  # Number of samples in each class for training

    total_samples = sum(class_samples)
    samples = total_samples/len(class_samples)
    class_weights = [samples / (s + 1e-8) for s in class_samples]
    class_weights = torch.tensor(class_weights)
    return class_weights


def get_model(device, image_dict: dict, model:str="swin"):
    torch.cuda.empty_cache()

    if model == "resnet":
        model_name = "timm/resnet18.a1_in1k"
        model_file = "models/Master_resnet.pth"
    else:
        model_name = 'swinv2_tiny_window8_256.ms_in1k'
        model_file = "models/Master_swin.pth"

    model = timm.create_model(
        model_name,
        pretrained=False,
        features_only=False,
        num_classes = 8,
        drop_path_rate=0.2
    )

    if os.path.isfile(model_file):
        checkpoint1 = torch.load(model_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint1['model_state_dict'])
    model = model.to(device)
    return model
