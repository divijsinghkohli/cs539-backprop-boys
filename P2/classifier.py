import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

#ImageNet color statistics: the pretrained model expects inputs normalized with these
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

#Reads classes_50.csv and convert to a pandas table
def load_classes(csv_path):
    #1. Read the file
    classes = pd.read_csv(csv_path, dtype=str)
    #2. convert class_idx to numbers (0-49)
    classes["class_idx"] = classes["class_idx"].astype(int)
    #3 Sort by class_idx
    classes = classes.sort_values("class_idx").reset_index(drop=True)
    #4. Check class_idx runs 0-49
    assert list(classes["class_idx"]) == list(range(len(classes))), (
        "class_idx is not 0-49")
    #5. Return the table
    return classes


def build_model(num_classes=50, pretrained=True):
    #1. Choosing the weights If using pretrained then use ResNet50's IMAGENET1K_V2 weights
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    #2. Create RestNet-50 (Pre-trained model)
    model = resnet50(weights=weights)
    #3. Replace the last layer
    model.fc = nn.Linear(model.fc.in_features, num_classes) #Input: 2048, Output: 50
    #4. Return the model
    return model


def set_trainable(model, phase):
    #phase 1: Only the new output layer learns
    #phase 2: fine tuning the final block of ResNet
    #1. Set to train mode
    model.train()
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    #2. Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    #3. Unfreeze the output layer (learns in both phases: 1 and 2)
    for p in model.fc.parameters():
        p.requires_grad = True
    #4. Phase 2: also unfreeze the last block (layer4)
    if phase == 2:
        for p in model.layer4.parameters():
            p.requires_grad = True
    #5. Frozen BatchNorm layers keep their ImageNet statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()
    #6. Count trainable and frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def train_transforms():
    #Augment the data (Rotation, Flipping, Blurring, Lighting Changes)
    return transforms.Compose([
        #1. Change Orientations
        transforms.RandomRotation(degrees=180, fill=(255, 255, 255)),
        #2. Change Framing (Off center)
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        #3. Mirror images
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #4. Lighting changes and random hue
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.5),
        transforms.RandomGrayscale(p=0.2),
        #5. Image blur, like a phone photo
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        #6. Convet image to tensor
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms():
    #Follow the same steps as train_transforms except No data augmentation.
    return transforms.Compose([
        #1. Resize to 224x224 
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #2. Convert Image to Tensor
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(weights_path, device="cpu"):
    #1. Read the saved file: either the weights alone or a version during training
    saved = torch.load(weights_path, map_location=device, weights_only=True)
    state = saved["model_state"] if "model_state" in saved else saved
    #2. Build an empty model with the same number of classes as the saved one
    num_classes = state["fc.weight"].shape[0]
    model = build_model(num_classes=num_classes, pretrained=False)
    #3. Copy the saved weights in, move to the device, and switch to evaluation mode
    model.load_state_dict(state)
    model.to(device) #default on cpu
    model.eval()
    return model


def predict(model, crops, classes, device="cpu", batch_size=64):
    #crops: a list of PIL images (any size), or a tensor of N x 3 x 224 x 224 already passed through eval_transforms()
    #Returns: part IDs, confidence, and probabilities
    model.eval()
    #1. Turn images into one tensor batch
    if isinstance(crops, Image.Image):
        crops = [crops]
    if isinstance(crops, torch.Tensor):
        batch = crops
    else:
        prep = eval_transforms()
        batch = torch.stack([prep(img.convert("RGB")) for img in crops]) if len(crops) else torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE)
    #2. Run the model in chunks, without tracking gradients (Gradients not used for predictions)
    probs = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            scores = model(batch[start:start + batch_size].to(device))
            probs.append(torch.softmax(scores, dim=1).cpu())
    probs = torch.cat(probs) if probs else torch.empty(0, len(classes))
    #3. Set the highest probability to be the prediction. Translate class index to part ID
    confidences, class_idx = probs.max(dim=1)
    part_ids = [classes["part_id_current"].iloc[i] for i in class_idx.tolist()]
    return part_ids, confidences.tolist(), probs.numpy()
