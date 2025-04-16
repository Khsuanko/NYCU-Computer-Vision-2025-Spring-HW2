import os
import torch
import json
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# === Load Model ===
def load_model(weight_path, num_classes, device, score_thresh=0.5, nms_thresh=0.5):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    # Set thresholds
    model.roi_heads.score_thresh = score_thresh   # Confidence threshold
    model.roi_heads.nms_thresh = nms_thresh

    model.to(device)
    model.eval()
    return model

# === Inference on Test Set ===
def run_inference(model, test_dir, device, output_json, output_csv):
    results = []
    recog_results = []

    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')],
                         key=lambda x: int(os.path.splitext(x)[0]))

    for filename in tqdm(image_files, desc="Running Inference"):
        image_id = int(os.path.splitext(filename)[0])
        path = os.path.join(test_dir, filename)
        image = Image.open(path).convert("RGB")
        tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            output = model([tensor])[0]

        digits = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if score >= model.roi_heads.score_thresh:
                x_min, y_min, x_max, y_max = box.tolist()
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                results.append({
                    "image_id": image_id,
                    "bbox": bbox,
                    "score": float(score),
                    "category_id": int(label)
                })
                digits.append((x_min, int(label)))  # use x_min for left-to-right sorting

        # Task 2: Sort by x_min and join digits
        if digits:
            digits.sort()
            number_str = ''.join(str((digit - 1) % 10) for _, digit in digits)
            recog_results.append({"image_id": image_id, "pred_label": number_str})
        else:
            recog_results.append({"image_id": image_id, "pred_label": -1})

    with open(output_json, "w") as f:
        json.dump(results, f)

    pd.DataFrame(recog_results).to_csv(output_csv, index=False)

# === Main ===
def main():
    test_dir = "data/test"
    weight_path = "fasterrcnn_best.pth"
    output_json = "pred.json"
    output_csv = "pred.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 11  # 10 digits (category_id 1–10) + background

    score_thresh = 0.7
    nms_thresh = 0.3

    model = load_model(weight_path, num_classes, device, score_thresh, nms_thresh)
    run_inference(model, test_dir, device, output_json, output_csv)
    print(f"Saved Task 1: {output_json}")
    print(f"Saved Task 2: {output_csv}")

if __name__ == "__main__":
    main()
