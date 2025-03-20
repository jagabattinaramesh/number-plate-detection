from fastapi import FastAPI, File, UploadFile
import shutil
import os
from app.model import CustomModel  # Change from "model" to "app.model"

# Initialize FastAPI app
app = FastAPI()

# Load the YOLO + OCR model
model = CustomModel("weights/best.pt")

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

@app.get("/")
def home():
    return {"message": "Number Plate Detection API is running"}

@app.post("/detect/")
async def detect_number_plate(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    input_path = f"output/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    results = model.infer(input_path)

    return {"detections": results}
