# 🎤 Parkinson's Voice Detection

This project, **Perkinson-Voice-Detection**, uses advanced **Machine Learning** and **Deep Learning (RandomForest + CNN)** models to detect the presence of **Parkinson’s Disease** from **audio inputs**.  
A simple and interactive **Flask web app** is provided for real-time predictions.

---

## 🚀 Features
- 🧩 Dual model approach – RandomForest & CNN for better accuracy  
- 🎧 Voice-based input support  
- 🌐 Flask-powered web interface  
- 📊 Probability-based output for diagnosis confidence  
- 💾 Easy to deploy and extend for research or clinical purposes  

---

## 📁 Project Structure :
Perkinson-Voice-Detection/
│
├── app.py # Main Flask backend
├── parkinsons_rf_model.pkl # RandomForest trained model
├── parkinsons_cnn_model.h5 # CNN trained model
├── requirements.txt # Required Python libraries
├── templates/ # HTML templates for web interface
├── static/ # CSS, JS, and media assets
├── data/ # Audio dataset or sample inputs (optional)
└── README.md # Project documentation




---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Zeeshan3h3/Perkinson-Voice-Detection.git
cd Perkinson-Voice-Detection

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux


3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py



Now open your browser and go to:
👉 http://localhost:5000/





Models used : RandomForestClassifier   and    Convolutional Neural Network (CNN)



Input: Audio file (voice sample)

Output: Probability score indicating likelihood of Parkinson’s Disease

## Exmaple OF Result
Prediction: Parkinson’s Detected
Confidence: 0.87




📊 Dataset

This project is trained using voice-based biomedical data containing various acoustic measures of Parkinson’s patients and healthy individuals.

📚 Dataset source: UCI Parkinson’s Disease Dataset : https://archive.ics.uci.edu/dataset/174/parkinsons



🧪 Future Improvements

Integrate real-time microphone input

Add model comparison dashboard (RF vs CNN)

Deploy using Render, Hugging Face Spaces, or Streamlit Cloud

Collect user feedback for adaptive learning





🪪 License

This project is open-source and available under the MIT License
.




💬 Contact

Developer: MD.Zeeshan

📧 Email: mdzeeshan08886@gmail.com

🌐 GitHub: https://github.com/Zeeshan3h3