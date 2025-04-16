# NYCU-Computer-Vision-2025-Spring-HW2  
StudentID: 110550122  
Name: 柯凱軒

## Introduction
This practice focuses on a digit recognition task, where the goal is twofold:  
Digit Detection: Detect individual digits within an image and output their bounding boxes in COCO format (Task 1).  
Digit Recognition: Infer the full number by recognizing and ordering detected digits from left to right (Task 2).  
  
To solve this, we adopted the Faster R-CNN architecture, a two-stage object detector that performs both region proposal and classification.  
This approach is well-suited for structured detection tasks like digit spotting due to its high accuracy and flexibility in modifying the backbone, neck, and head components.

## How to install
1. Install Dependencies  
```python
pip install torch torchvision pandas tqdm
```
2. Ensure you have the dataset structured as follows:
```python
./data/
    ├── train/
    ├── val/
    ├── test/
    ├── train.json
    ├── valid.json
```
3. Run the code
```python
python train.py
python evaluate.py
```
## Performance snapshot  
![performance](https://github.com/Khsuanko/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/performance.png)
