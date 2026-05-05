import tensorflow as tf
import os

# Point this to your PROCESSED training data
data_path = "C:/BEProject/data_processed/train"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    batch_size=32,
    label_mode='categorical'
)


print("\n--- CRITICAL: MODEL LABEL MAPPING ---")
for i, name in enumerate(train_ds.class_names):
    print(f"Internal Index {i}  ==>  Is Folder Name: {name}")