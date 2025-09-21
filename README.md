# Bird Species Prediction  

This project builds a **Convolutional Neural Network (CNN)** model to classify bird species from images. The dataset is loaded from Google Drive, processed, and trained using deep learning libraries such as **Keras** and **TensorFlow**.  

## Features  
- Loads dataset from Google Drive.  
- Uses **OpenCV** and **Matplotlib** for image preprocessing and visualization.  
- Encodes species labels using `LabelBinarizer`.  
- Builds a CNN with layers: `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`.  
- Trains the model on bird images and evaluates accuracy.  
- Visualizes training history and predictions.  

## Requirements  
The notebook requires the following Python libraries:  
```bash
numpy  
pandas  
matplotlib  
opencv-python  
scikit-learn  
tensorflow  
keras  
```

Install missing dependencies in Colab with:  
```bash
!pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow keras
```

## Dataset  
The dataset must be placed in Google Drive, structured as:  
```
/content/drive/My Drive/Bird Species Prediction/Data/Bird Speciees Dataset/
    ├── Species_1
    ├── Species_2
    ├── ...
```

Each subfolder should contain images of a bird species.  

## Usage  

### 1. Mount Google Drive  
```python
from google.colab import drive
drive.mount("/content/drive")
```

### 2. Import Libraries  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2, random
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
```

### 3. Load and Preprocess Images  
```python
data, labels = [], []
path = "/content/drive/My Drive/Bird Species Prediction/Data/Bird Speciees Dataset/"

for folder in listdir(path):
    folder_path = path + folder
    for img in listdir(folder_path):
        image = cv2.imread(folder_path + '/' + img)
        image = cv2.resize(image, (128, 128))  # Resize images
        data.append(img_to_array(image))
        labels.append(folder)

# Normalize
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Train-test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)
```

### 4. Build CNN Model  
```python
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(lb.classes_), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
```

### 5. Train the Model  
```python
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=32)
```

### 6. Evaluate Performance  
```python
loss, acc = model.evaluate(testX, testY)
print(f"Test Accuracy: {acc*100:.2f}%")
```

### 7. Plot Training History  
```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

## Output  
- Trained CNN model for bird species classification.  
- Visualization of dataset samples.  
- Training and validation accuracy/loss plots.  
- Prediction results on test images.  

## Future Improvements  
- Apply **data augmentation** (`ImageDataGenerator`) to improve robustness.  
- Experiment with deeper architectures (ResNet, VGG16).  
- Deploy model as a **Flask/Django web app** or **mobile app** for real-time bird identification.  
