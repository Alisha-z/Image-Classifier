# Image-Classifier
# 🐱🐶 Cats vs Dogs Image Classification System with GUI

This project is a **Deep Learning-based Image Classifier** that can distinguish between images of **cats** and **dogs** using a **Convolutional Neural Network (CNN)**.  
It includes a **Tkinter-based Graphical User Interface (GUI)** for real-time user interaction, allowing users to upload images and get instant predictions.

---

## 📌 Features

### 🔹 Model Features
- Trains a **CNN** using the `cats_and_dogs_filtered` dataset from TensorFlow.
- Uses **Image Augmentation** (rotation, zoom, flipping) to improve accuracy.
- Supports both **training from scratch** and **loading a pre-trained model**.
- Achieves high accuracy on the validation dataset.
- Saves the trained model in `.h5` format for later use.

### 🔹 GUI Features
- **Dark mode** and **modern layout** for a clean look.
- **Upload button** to select an image from your computer.
- **Real-time classification** into **Cat** or **Dog** with confidence score.
- **Image preview** inside the application window.
- **Prediction logs** displayed inside the GUI.
- **Exit button** to close the application safely.

## Project Structure 
Cats-vs-Dogs-Classifier/
│
├── Image_classifier.py # Script to train the CNN model
├── Image_classifier_GUI.py # Tkinter GUI for image classification
├── model.h5 # Saved trained model (created after training)
├── README.md # Project documentation
├── requirements.txt # Required dependencies
│
└── data/
├── train/ # Training images (cats & dogs)
│ ├── cats/
│ └── dogs/
│
└── validation/ # Validation images (cats & dogs)
├── cats/
└── dogs/


## 🛠 Requirements

Install the dependencies using:
```bash
pip install -r requirements.txt
requirements.txt

nginx
Copy
Edit
tensorflow
keras
numpy
matplotlib
Pillow
tkintertable

How to Use
1️⃣ Train the Model
Run the training script:

bash
Copy
Edit
python train.py
This will:

Load training and validation data.

Apply data augmentation.

Train a CNN model.

Save the model as model.h5.

2️⃣ Run the GUI
After training, launch the GUI.

Steps inside GUI:
Click "Upload Image".

Select an image of a cat or dog.

The system will display:
The uploaded image.

📊 Training Details
Optimizer: Adam

Loss Function: Binary Crossentropy

Batch Size: 32

Epochs: 20

Metrics: Accuracy

---

Predicted class (Cat or Dog).

Confidence percentage.

🧠 Model Architecture
The CNN model contains:

Conv2D layers with ReLU activation.

MaxPooling2D for downsampling.

Flatten layer for vector conversion.

Dense layers for classification.

Sigmoid activation in the output layer (binary classification).


---

## 📂 Project Structure

