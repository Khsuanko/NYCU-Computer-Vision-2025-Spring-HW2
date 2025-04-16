import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from pycocotools.coco import COCO
from collections import defaultdict
from tqdm import tqdm

# === Dataset ===
class DigitDataset(Dataset):
    def __init__(self, image_dir, json_path, is_train=False):
        self.image_dir = image_dir
        self.coco = COCO(json_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.is_train = is_train
        
        # Training augmentations
        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])
        
        # Validation/inference transform (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        # Apply appropriate transform
        if self.is_train:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
            
        return image, target

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# === Model ===
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# === Training Loop ===
def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        
        # Optional: Step per batch (for some schedulers)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()
    
    return running_loss / len(data_loader)

# === Validation mAP (IoU > 0.5) Approximation ===
def evaluate(model, data_loader, device):
    model.eval()
    detected = defaultdict(list)
    gt = defaultdict(list)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                gt[image_id].extend(target["labels"].tolist())
                pred = output["labels"][output["scores"] > 0.5].tolist()
                detected[image_id].extend(pred)

    correct, total = 0, 0
    for img_id in gt:
        correct += len(set(gt[img_id]) & set(detected[img_id]))
        total += len(set(gt[img_id]))
    return correct / total if total > 0 else 0

# === Main ===
def main():
    # Paths
    train_dir = "data/train"
    valid_dir = "data/valid"
    train_json = "data/train.json"
    valid_json = "data/valid.json"
    output_path = "fasterrcnn_best.pth"

    # Hyperparameters
    num_classes = 11  # 10 digits (category_id 1–10) + background
    num_epochs = 40
    lr = 0.005
    batch_size = 4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Dataloader
    train_dataset = DigitDataset(train_dir, train_json)
    valid_dataset = DigitDataset(valid_dir, valid_json)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model, optimizer
    model = get_model(num_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)

     # Learning rate scheduler (Cosine with warm restarts)
#    scheduler = CosineAnnealingWarmRestarts(
#        optimizer,
#        T_0=5,          # Number of epochs for first restart cycle
#        T_mult=1,       # Factor to increase cycle length after restart
#        eta_min=1e-5    # Minimum learning rate
#    )

    # Cosine Annealing LR Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # Maximum number of iterations (epochs)
        eta_min=1e-5       # Minimum learning rate
    )

    # Train & Eval
    best_score = 0
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)

        # Step the scheduler (per epoch)
        scheduler.step()
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} loss: {loss:.4f} | LR: {current_lr:.6f}")

        score = evaluate(model, valid_loader, device)
        print(f"Validation score (IoU@0.5 approx.): {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model with score {score:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
