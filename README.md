# RetinAI

## Description and Application

**RetinAI** is an advanced system specialized in the early detection and classification of eye diseases by analyzing retinal images in combination with basic patient demographic data (gender and age).

The primary goal of this agent is to support ophthalmologists and medical professionals in the diagnostic process, enabling faster, more accurate, and standardized identification of conditions such as:

- Glaucoma  
- Diabetic Retinopathy  
- Cataracts  
- Age-related Macular Degeneration  
- Other ocular abnormalities

Due to the technical challenges in obtaining high-quality retinal images outside of medical facilities, **RetinAI is designed as a professional diagnostic tool for clinical use**. Its implementation helps:

- Enable earlier medical interventions  
- Reduce the risk of permanent vision damage  
- Optimize time and resource management in clinical practice

Additionally, the system supports **systematic analysis and trend monitoring** of eye diseases across larger patient populations through integrated visual statistics.

---

## How the Agent Learns

At the core of the system is a deep learning model based on the **EfficientNetB0** architecture, a convolutional neural network (CNN) optimized for high efficiency and image classification accuracy.

The model is trained on a specialized dataset containing **over 5,000 patient records**, including:

- Left and right retina images  
- Associated demographic information (age, gender)  
- Diagnostic labels

A secondary model, based on the **MobileNetV2** architecture, is integrated for **image input validation**, ensuring only valid retina images are processed — this helps prevent inaccurate predictions.

### Preprocessing & Training

Input data goes through a comprehensive preprocessing pipeline:

- Image scaling to appropriate dimensions  
- Merging into a unified representation  
- Combining with demographic data (age and gender)

Model training includes:

- **Adam Optimizer**  
- **EarlyStopping** (to prevent overfitting)  
- **ModelCheckpoint** (to save the best-performing models)

### Continuous Learning

RetinAI includes an **automated retraining mechanism**, triggered after every 10 valid predictions. This enables:

- Continuous learning  
- Adaptation to new data patterns  
- Maintaining system relevance and accuracy over time

---

## Features

- ✅ Validation of retinal images and patient demographic data  
- 🔍 Prediction of eye diseases with percentage-based probability per category  
- 📊 Graph-based visualization of disease frequency  
- 🔁 Automatic retraining to adapt to new data and improve accuracy
