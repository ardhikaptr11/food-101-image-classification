# Food-101 Image Classification

![Python](https://img.shields.io/badge/Python-3.12.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.17.0-FF6F00)
![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow)

A deep learning pipeline using EfficientNetV2B2 (transfer learning + fine-tuning) to classify food images from a 10-class subset of Food-101.

## Dataset

This project uses the [Food-101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) dataset, which contains 101,000 images across 101 food categories. Due to compute and time constraints (and per the assignment requirements), training was limited to a randomly selected subset of **10 classes**:

- Beef Carpaccio
- Caprese Salad
- Carrot Cake
- Cheesecake
- Croque Madame
- Donuts
- Escargots
- Ramen
- Sashimi
- Strawberry Shortcake

A preview grid sampling one random image per class is generated in the notebook to sanity-check the data before training.

### Data Splits

The official Food-101 train/test split was used as the base, with the training portion further split into training and validation sets using a stratified 80/20 split (`random_state=42`) to preserve class balance.

| Split | Images |
|---|---|
| Train | 6,000 |
| Validation | 1,500 |
| Test | 2,500 |

A verification step confirms all 10 classes are represented in every split before proceeding.

## Approach / Methodology

### Preprocessing & Augmentation

Images are decoded, resized to `224x224`, and batched using `tf.data` pipelines (with `AUTOTUNE` for prefetching). For the training set, the following augmentations are applied on the fly:

- Random horizontal and vertical flips
- Random brightness adjustment
- Random contrast adjustment
- Random saturation adjustment
- Resize-then-random-crop (resizing to a slightly larger size before cropping back to `224x224`, to introduce spatial variance)

Pixel values are clipped to a valid `[0, 255]` range after augmentation.

### Model Architecture

The model uses **EfficientNetV2B2** (pretrained on ImageNet) as a frozen feature extractor, with a custom classification head on top:

1. `EfficientNetV2B2` base (`include_top=False`, frozen weights)
2. `Conv2D` — 512 filters, 3x3 kernel, ReLU activation, L2 regularization
3. `BatchNormalization`
4. `GlobalAveragePooling2D`
5. `Dropout` (rate 0.6)
6. `Dense` output layer — softmax activation, L2 regularization, one unit per class

### Training Strategy

Training is done in two phases:

**Phase 1 — Feature Extraction**
- Base model frozen; only the custom head is trained
- Optimizer: Adam, learning rate `1e-3`
- 10 epochs

**Phase 2 — Fine-Tuning**
- Base model unfrozen except for its first layers (last 10 layers made trainable)
- Optimizer: Adam, learning rate `1e-4`
- Additional 15 epochs (continuing from Phase 1)

Both phases share the same callbacks:
- **EarlyStopping** — monitors validation accuracy, patience of 6, restores best weights
- **ModelCheckpoint** — saves the best-performing model based on validation accuracy
- **ReduceLROnPlateau** — reduces learning rate by a factor of 0.2 after 2 epochs of no improvement (down to a minimum of `1e-6`)

## Results

Final model performance after both training phases:

| Split | Accuracy | 
|---|---|
| Train | 98.97% |
| Validation | 90.47% |
| Test | 92.24% |

Training and validation accuracy/loss curves are plotted across both phases, with a marker indicating where fine-tuning begins, to visualize how the model's performance evolved.

A qualitative check is also included: a grid of sample test predictions showing the actual vs. predicted label for each image, color-coded green for correct and red for incorrect predictions.

## Model Export & Inference

After training, the model is saved in TensorFlow's `SavedModel` format, then converted into two additional formats:

- **TensorFlow Lite (`.tflite`)** — converted directly from the SavedModel using `TFLiteConverter`, with default optimizations applied.
- **TensorFlow.js** — converted from the SavedModel using the `tensorflowjs_converter` CLI tool.

The TensorFlow.js conversion required a workaround: the notebook's default Python 3.12 environment removed legacy components (`distutils`, `np.object`) that the TensorFlow.js ecosystem depends on. To resolve this, a separate Python 3.10 virtual environment was created specifically for running the conversion, using pinned versions of TensorFlow, TensorFlow.js, NumPy, and protobuf.

## Tech Stack

**Modeling**
- TensorFlow / Keras
- EfficientNetV2B2 (pretrained on ImageNet)

**Data Handling**
- NumPy
- scikit-learn (train/validation split)
- Pillow (PIL)

**Visualization**
- Matplotlib

**Model Conversion**
- TensorFlow Lite
- TensorFlow.js

**Environment**
- Google Colab
- uv (package management)

## How to Run

This notebook is designed to run on Google Colab.

1. Download the [Food-101 dataset](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) and upload the `.tar.gz` file to your Google Drive.
2. Update the `target_file` path in the notebook to match where you placed the dataset in your Drive (it's currently hardcoded to a personal path).
3. Open the notebook in Colab and run the cells in order.

## Author

**I Putu Crisna Putra Ardhika**
- Email: putucrisna11@gmail.com
- Github: [ardhikaptr11](https://github.com/ardhikaptr11)