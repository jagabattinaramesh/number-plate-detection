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
You can find the Jupyter Notebook for merging these data sets [here](src/code_for_merging.ipynb).



## Train YOLO model
```bash
# Inside VM
# Build and Run the Docker Container
cd ultralytics

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
## Executing Detection & OCR  models
Build the docker image using [Dockerfile](/Dockerfile).
By default it installs dependencies present in [requirements file](/requirements.txt) and executes YOLO and OCR models.

### Building docker
```bash
docker build -t number-plate-api .
```
### Run the docker and deploy using FastAPI
```bash
docker run --rm -p 8000:8000 -v $(pwd)/output:/app/output number-plate-api
```
### Check the deployed model in any browser
http://127.0.0.1:8000/docs

</div>
