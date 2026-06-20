```markdown
# 🧠 Brain Tumor Classification using VGG16

A deep learning project that automatically classifies **Brain MRI images** into different tumor categories using **Transfer Learning** with the **VGG16 Convolutional Neural Network (CNN)**. This project leverages image preprocessing, augmentation, and a pre-trained VGG16 model to improve classification performance on medical imaging data.

---

## 📖 Overview

Brain tumors are among the most serious neurological disorders, and early diagnosis is critical for effective treatment. Manual interpretation of MRI scans is time-consuming and requires expert radiologists.

This project demonstrates how **Transfer Learning** can be used to build an efficient brain tumor classification model using the **VGG16** architecture. The model is trained on MRI images after applying preprocessing and data augmentation techniques to improve accuracy and generalization.

---

## 🎯 Objectives

- Classify brain MRI images into multiple tumor categories.
- Utilize the pre-trained VGG16 model for feature extraction.
- Improve model performance using image augmentation.
- Build an efficient and accurate deep learning classifier.

---

## 🚀 Features

- ✅ Transfer Learning using VGG16
- ✅ MRI Image Classification
- ✅ Multi-Class Classification
- ✅ Image Preprocessing
- ✅ Data Augmentation
- ✅ Image Normalization
- ✅ Automatic Label Encoding
- ✅ Batch Data Generation
- ✅ Model Training & Validation
- ✅ Save Trained Model

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

```

Brain-Tumor-Classification/
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
├── models/
│   └── brain_tumor_vgg16.h5
│
├── notebooks/
│   └── Brain_Tumor_Classification.ipynb
│
├── images/
│   └── sample_predictions.png
│
├── requirements.txt
├── README.md
└── main.py

````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Brain-Tumor-Classification.git
````

Navigate to the project directory:

```bash
cd Brain-Tumor-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the training script:

```bash
python main.py
```

Or open the Jupyter Notebook:

```bash
jupyter notebook
```

---

## 🧪 Image Preprocessing

The following preprocessing steps are applied:

* Resize MRI images
* Normalize pixel values
* Brightness augmentation
* Contrast enhancement
* Data augmentation
* Label encoding

---

## 🧠 Model Architecture

The model is built using the **VGG16** pre-trained network.

* Pre-trained VGG16 (ImageNet weights)
* Global Average Pooling
* Fully Connected Dense Layers
* Dropout Layer
* Softmax Output Layer

---

## 📊 Training

The model is trained using:

* Categorical Crossentropy Loss
* Adam Optimizer
* Mini-batch Gradient Descent
* Validation Split
* Early Stopping (Optional)

---

## 📈 Results

The trained model is capable of accurately classifying MRI images into their respective brain tumor classes.

Example evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 💾 Model Saving

After training, the model is saved for future inference:

```python
model.save("brain_tumor_vgg16.h5")
```

---

## 🔮 Future Improvements

* Fine-tune the VGG16 base model
* Deploy using Flask or Streamlit
* Add Grad-CAM visualization
* Hyperparameter optimization
* Improve dataset size and diversity

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/your-username

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub!

```
```
