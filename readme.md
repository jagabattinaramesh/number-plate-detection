<div align="center">

# Number plate detection using YOLO and OCR

In this project image of any vehicle is uploaded to the model and it will detect alpha numeric values in the number plate.
Here to detect the number plate we have used YOLO object detection model and to find letters in it paddleocr is implemented. Later FastAPI is used to deploy the model.

</div>

<div align="center">

## Implementation

</div>

<div align="left">

## Data Handling

### Data
| S.No | Data Source |Number of Images | Url |
|----------|----------|----------|----------|
| 1 | Kaggle | 433 | [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data) |
| 2 | Roboflow |  545 | [UK Number Plate Recognision](https://universe.roboflow.com/recognision-datasets/uk-number-plate-recognision/dataset/2) |
| 3 | Roboflow |  8823 | [Vehicle Registration Plates](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/1) |

### Script for merging the data
```python

import json
import shutil
import os

def merge_coco_datasets(json_paths, image_dirs, dataset_names, output_json, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)  # Ensure output directory exists
    
    merged_data = {"images": [], "annotations": [], "categories": []}
    image_id_offset = 0
    annotation_id_offset = 0
    category_map = {}  # Maps old category IDs to new ones

    for json_path, img_dir, dataset_name in zip(json_paths, image_dirs, dataset_names):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Remap categories (if not already added)
        if not merged_data['categories']:
            merged_data['categories'] = data['categories']
            category_map = {c['id']: c['id'] for c in data['categories']}
        else:
            for c in data['categories']:
                if c['id'] not in category_map:
                    new_cat_id = len(merged_data['categories']) + 1
                    c['id'] = new_cat_id
                    category_map[c['id']] = new_cat_id
                    merged_data['categories'].append(c)

        img_id_map = {}  # Maps old image IDs to new ones
        for img in data['images']:
            old_id = img['id']
            new_filename = f"{dataset_name}_{img['file_name']}"  # Rename image uniquely
            new_path = os.path.join(output_image_dir, new_filename)

            # Copy and rename the image
            shutil.copy(os.path.join(img_dir, img['file_name']), new_path)

            img['id'] = image_id_offset
            img['file_name'] = new_filename  # Update file name in JSON
            img_id_map[old_id] = img['id']
            merged_data['images'].append(img)
            image_id_offset += 1  # Increment image ID

        for ann in data['annotations']:
            ann['id'] = annotation_id_offset
            ann['image_id'] = img_id_map[ann['image_id']]  # Update annotation to new image ID
            ann['category_id'] = category_map[ann['category_id']]  # Update category IDs
            merged_data['annotations'].append(ann)
            annotation_id_offset += 1  # Increment annotation ID

    # Save merged annotations
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=4)

# Example usage
merge_coco_datasets(
    json_paths=["kaggle_train_coco.json", "uk_train_coco.json", "v_train_coco.json"],
    image_dirs=["kaggle_train_images", "uk_train_images", "v_train_images"],
    dataset_names=["kaggle", "uk", "v"],  # Prefix for unique naming
    output_json="merged_annotations.json",
    output_image_dir="merged_images"
)
```
</div>


### Train YOLO model
```bash
# Inside VM
# Build and Run the Docker Container

docker build -t ultralytics-yolo -f docker/Dockerfile .

docker run --gpus all --rm -it -v $(pwd):/usr/src/ultralytics/data ultralytics_yolo bash

# Run YOLO Training
python ultralytics/cfg/train.py \
    --model yolov8n.pt \
    --data /path/to/dataset.yaml \
    --epochs 200 \
    --imgsz 640 \
    --device 0

# Retrieve the Best Model
/runs/detect/train/weights/best.pt
```

