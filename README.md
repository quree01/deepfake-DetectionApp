# 🎭 Deepfake Detection App

An end-to-end Deepfake Detection System built with Streamlit, leveraging advanced deep learning models to detect manipulated audio and video content with high accuracy.

---

## 📌 Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 About

This project aims to address the growing threat of deepfake media by providing an easy-to-use web interface for detecting manipulated audio and video files.

Built with:

- 🐍 Python
- ⚙️ TensorFlow / Keras
- 📊 Streamlit for deployment

---

## 🚀 Features

✅ Detects video deepfakes using ResNet  
✅ Detects audio deepfakes using VGG16 + LSTM  
✅ MFCC feature extraction for robust audio detection  
✅ Clean Streamlit interface for easy uploads and results  
✅ Supports deployment on local server or cloud

---

## 📂 Project Structure

```
deepfake-DetectionApp/
│
├── models/
│   ├── video_resnet_model.kera
│   ├── audio_vgg16_lstm_model.kera
│
├── src/
│   ├── audio_processing.py
│   ├── video_processing.py
│   ├── streamlit_app.py
│
├── images/
│   └── app_screenshot.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🗂️ Datasets

- **Video:** FaceForensics++  
- **Audio:** Fake or Real Audio Dataset

---

## 🧹 Preprocessing

- **Video:** Standard face extraction, frame resizing, normalization.  
- **Audio:** MFCC feature extraction, normalization, fixed length 2s clips.

---

## 🧠 Models

- 🎞️ **Video Model:** ResNet-based deep CNN trained on FaceForensics++ dataset.
- 🎙️ **Audio Model:** VGG16 feature extractor layered with LSTM for temporal analysis, trained on Fake or Real dataset.

---

## ⚙️ Installation

1️⃣ **Clone the repository**

```bash
git clone https://github.com/quree01/deepfake-DetectionApp.git
cd deepfake-DetectionApp
```

2️⃣ **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🏃 Usage

1️⃣ **Run the Streamlit app**

```bash
streamlit run src/streamlit_app.py
```

2️⃣ **Open in your browser**

Streamlit will open the app automatically at [http://localhost:8501](http://localhost:8501)

3️⃣ **Upload audio or video**

- Upload a file (mp4, wav, etc.)
- Click **Predict**
- View detection result: **REAL** or **FAKE**

---

## 🔗 Demo

**Live App:** [Releasing Soon]

**Screenshots:**  
![Screenshot 1](https://github.com/user-attachments/assets/95ae01b7-0e2f-40a5-aa5a-8f94bc28e1db)
![Screenshot 2](https://github.com/user-attachments/assets/9e08fa30-0943-4048-8b47-b279abe09d0c)
![Screenshot 3](https://github.com/user-attachments/assets/8a1d2fc6-c97b-4750-98a5-43ca9030a72f)

---

## 📊 Results

| Model              | Dataset           | Accuracy |
|--------------------|-------------------|----------|
| ResNet (Video)     | FaceForensics++   | 91%      |
| VGG16 + LSTM (Audio) | Fake or Real      | 89%      |

✅ **Metrics:** Accuracy, Confusion Matrix, Precision, Recall.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or feature suggestions.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

**Author:** quree01
