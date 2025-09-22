from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn

# load your trained model
model = load_model("bird_species.h5")  # ensure the .h5 is in the repo root

# adjust this if you trained with a different input size
IMG_SIZE = (224, 224)

app = FastAPI()

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
