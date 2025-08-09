import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Paths to your dataset
train_dir = r"C:\Users\Alisha\.keras\datasets\cats_and_dogs_extracted\cats_and_dogs_filtered\train"
validation_dir = r"C:\Users\Alisha\.keras\datasets\cats_and_dogs_extracted\cats_and_dogs_filtered\validation"

# Data preprocessing & augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Build model
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train (feature extraction)
initial_epochs = 5
history = model.fit(train_generator, epochs=initial_epochs, validation_data=validation_generator)

# Fine-tune (unfreeze last layers)
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(train_generator, epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_generator)

# Combine history for visualization
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Visualization: Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Visualization: Sample predictions
class_names = list(train_generator.class_indices.keys())
for data_batch, labels_batch in validation_generator:
    preds = (model.predict(data_batch) > 0.5).astype("int32")
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(data_batch[i])
        true = class_names[int(labels_batch[i])]
        pred = class_names[preds[i][0]]
        plt.title(f"True: {true}\nPred: {pred}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    break

# Confusion Matrix Visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = []
y_pred = []
for data_batch, labels_batch in validation_generator:
    preds = (model.predict(data_batch) > 0.5).astype("int32").flatten()
    y_true.extend(labels_batch.astype("int32"))
    y_pred.extend(preds)
    if len(y_true) >= validation_generator.samples:
        break

cm = confusion_matrix(y_true[:validation_generator.samples], y_pred[:validation_generator.samples])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Save model
model.save("cats_dogs_mobilenetv2_finetuned.keras")

# Webcam Real-Time Prediction
import cv2

print("Starting webcam for real-time prediction. Press 'q' to quit.")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (150, 150))
    img_norm = img / 255.0
    img_exp = np.expand_dims(img_norm, axis=0)
    pred = (model.predict(img_exp) > 0.5).astype("int32")[0][0]
    label = class_names[pred]
    cv2.putText(frame, f'Predicted: {label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Webcam - Cats vs Dogs', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()