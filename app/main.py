from fastapi import FastAPI, Request, HTTPException
import base64
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
import gc
import tensorflow as tf
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  
    allow_methods=['*'],  
    allow_headers=['*']   
)

# Load class names for bird classification
with open('bird_list_classes.json', 'r') as file:
    classes = json.load(file)

size = 299
img_size = (size, size)

# Load TFLite model
with open('40classesModel.tflite', 'rb') as f:
    tflite_model = f.read()

# Convert TFLite model to interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

def base64_to_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        img_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def process_image(base64_input):
    try:
        # Decode the image from base64
        img = base64_to_image(base64_input)
        
        # Detect birds using YOLOv8
        results = yolo_model(img)
        
        crop_results = []
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes
            
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
                
                # Crop the detected region from the image
                cropped_img = img[y_min:y_max, x_min:x_max]
                cropped_img_resized = cv2.resize(cropped_img, img_size)
                
                # Prepare the image for classification
                img_array = np.expand_dims(cropped_img_resized, axis=0).astype('float32') / 255.0
                
                # Perform classification with TFLite model
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                probabilities = list(predictions[0])
                threshold = 0.4
                isBird = bool(np.max(probabilities) >= threshold)  # Convert to Python bool
                
                if isBird:  # Check if the detected object is likely a bird
                    sorted_indexes = np.argsort(probabilities)[::-1][:5]
                    classes_id = []
                    predict_rank = []
                    probability_rank = []
                    for i in sorted_indexes:
                        class_id = str(i)
                        class_name = classes.get(class_id, "Unknown")
                        probability = probabilities[i]
                        classes_id.append(class_id)
                        predict_rank.append(class_name)
                        probability_rank.append(probability.item())
                    
                    crop_results.append({
                        "crop_boundaries": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        },
                        "predictions": [
                            {"rank": 1, "class_id": classes_id[0], "class_name": predict_rank[0], "probabilities": probability_rank[0]},
                            {"rank": 2, "class_id": classes_id[1], "class_name": predict_rank[1], "probabilities": probability_rank[1]},
                            {"rank": 3, "class_id": classes_id[2], "class_name": predict_rank[2], "probabilities": probability_rank[2]},
                            {"rank": 4, "class_id": classes_id[3], "class_name": predict_rank[3], "probabilities": probability_rank[3]},
                            {"rank": 5, "class_id": classes_id[4], "class_name": predict_rank[4], "probabilities": probability_rank[4]}
                        ],
                        "isBird": isBird
                    })
        
        # Explicitly call garbage collection
        gc.collect()
        
        return crop_results  # Return as a list without any keys
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"Hello": "This is the Bird Classification API"}

@app.post("/api/birdClassify")
async def read_image(image_data: Request):
    try:
        image_data_json = await image_data.json()
        base64_input = image_data_json.get("image")
        if not base64_input:
            raise HTTPException(status_code=400, detail="No image data found")
        
        result = process_image(base64_input)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
