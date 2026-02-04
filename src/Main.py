import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# CONFIGURATION
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 10
EPOCHS_VGG = 5

TRAIN_DIR = "C:\\Users\\amith\\Downloads\\11\\monkey-species-cnn-transfer-learning1\\data\\Training_Data"
TEST_DIR = "C:\\Users\\amith\\Downloads\\11\\monkey-species-cnn-transfer-learning1\\data\\Prediction_Data"
ERROR_DIR = "C:\\Users\\amith\\Downloads\\11\\monkey-species-cnn-transfer-learning1\\data\\test_error_images"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# DATA GENERATORS
# ---------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes

# ---------------------------
# CNN MODEL 1
# ---------------------------
def build_cnn_model_1():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------------------
# CNN MODEL 2 (WITH DROPOUT)
# ---------------------------
def build_cnn_model_2():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------------------
# TRAIN BOTH CNNs
# ---------------------------
print("\nTraining CNN Model 1...")
cnn1 = build_cnn_model_1()
cnn1.fit(train_gen, epochs=EPOCHS_CNN)
loss1, acc1 = cnn1.evaluate(test_gen)

print("\nTraining CNN Model 2...")
cnn2 = build_cnn_model_2()
cnn2.fit(train_gen, epochs=EPOCHS_CNN)
loss2, acc2 = cnn2.evaluate(test_gen)

# ---------------------------
# SELECT BEST CNN
# ---------------------------
best_model = cnn1 if acc1 >= acc2 else cnn2
best_model.save(f"{MODEL_DIR}/best_model.keras")
print(f"\nBest CNN saved (accuracy={max(acc1, acc2):.4f})")

# ---------------------------
# CONFUSION MATRIX (BEST CNN)
# ---------------------------
y_true = test_gen.classes
y_pred = np.argmax(best_model.predict(test_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=list(train_gen.class_indices.keys()))

plt.figure(figsize=(10,8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Best CNN")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_best_cnn.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------
# VGG16 TRANSFER LEARNING
# ---------------------------
print("\nTraining VGG16 Transfer Learning Model...")

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

vgg_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

vgg_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

vgg_model.fit(train_gen, epochs=EPOCHS_VGG)
vgg_model.save(f"{MODEL_DIR}/tuned.keras")

# ---------------------------
# CONFUSION MATRIX (VGG16)
# ---------------------------
y_pred_vgg = np.argmax(vgg_model.predict(test_gen), axis=1)
cm_vgg = confusion_matrix(y_true, y_pred_vgg)

disp = ConfusionMatrixDisplay(cm_vgg, display_labels=list(train_gen.class_indices.keys()))
plt.figure(figsize=(10,8))
disp.plot(cmap="Greens", xticks_rotation=45)
plt.title("Confusion Matrix - VGG16")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_vgg16.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------
# TASK 3: ERROR ANALYSIS
# ---------------------------
print("\nError Analysis on test_error_images:")

for img_name in os.listdir(ERROR_DIR):
    img_path = os.path.join(ERROR_DIR, img_name)
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = vgg_model.predict(arr)
    pred_class = list(train_gen.class_indices.keys())[np.argmax(pred)]

    print(f"{img_name} → predicted as: {pred_class}")

print("\nALL TASKS COMPLETED SUCCESSFULLY.")
