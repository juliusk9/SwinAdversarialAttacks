# utils
import os
import shutil
import json
import glob

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
    
    def __getname__(self, index):
        return self.image_list[index].split(os.sep)[-1]
    
    def __getzoom__(self, index):
        return self.image_list[index].split("_")[-1].split("-")[3]
    
    def __getclass__(self, index):
        return self.image_list[index].split("_")[-1].split("-")[0]

    def __len__(self):
        return self.data_len
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        ce_loss = nn.BCELoss()(probs, labels)
        weight = (1 - probs).pow(self.gamma)
        loss = ce_loss  # Initialize loss with cross-entropy loss
        if self.class_weights is not None:
            weight = weight * self.class_weights
            loss = loss * weight
        return loss
    
    
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
                albumentations.Normalize(mean=(0.787, 0.625, 0.765),
                                std=(0.105, 0.138, 0.089), p=1),
                albumentations.pytorch.transforms.ToTensorV2()
            ]),

            'valid': albumentations.Compose([
                albumentations.Resize(256, 256),
                albumentations.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(1.0, 1.0, 1.0), p=1),
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
        print("Opening Train.txt")
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


def get_model(device, image_dict: dict):
    torch.cuda.empty_cache()

    model = timm.create_model(
        'swinv2_tiny_window8_256.ms_in1k',
        pretrained=False,
        features_only=False,
        num_classes = 8,
        drop_path_rate=0.2
    )

    class_weights = get_class_weigths(image_dict).to(device)
    criterion = FocalLoss(class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    checkpoint1 = torch.load('Master.pth', map_location=torch.device(device))
    model.load_state_dict(checkpoint1['model_state_dict'])
    model = model.to(device)
    return model

