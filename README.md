# 😏 Sarcasm Detection Web Application

A Flask-based web app that detects sarcasm in text using machine learning. It provides real-time sarcasm classification, performance visualizations, and a modern UI. This project also includes a full dataset and detailed report.

---

## 📁 Project Structure

```
sarcasm-detector/
├── app.py
├── requirements.txt
├── Sarcasm_Headlines_Dataset.json
├── templates/
│   ├── index.html
│   └── visualizations.html
├── docs/
│   └── report.docx
```

---

## 🚀 Features

- Real-time sarcasm detection
- Confidence score output
- Text-to-speech result
- Statement history tracking
- Font selector and visual UI effects
- Model training with SMOTE and hyperparameter tuning
- Visualizations: Accuracy, Confusion Matrix, ROC

---

## 🔧 Setup Instructions

### 1. Clone and Install
```bash
git clone https://github.com/YOUR_USERNAME/sarcasm-detector.git
cd sarcasm-detector
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 🌍 Deploy Online (Render)

1. Go to [https://render.com](https://render.com)
2. Click **New Web Service**
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Deploy

---

## 🔁 GitHub Push Instructions

```bash
git init
git add .
git commit -m "Initial commit: sarcasm detector"
git remote add origin https://github.com/YOUR_USERNAME/sarcasm-detector.git
git branch -M main
git push -u origin main
```

---

## 📄 Project Report

📥 [Click to view report](docs/report.docx)

---

## 📊 API Endpoints

| Method | Route              | Description                      |
|--------|--------------------|----------------------------------|
| GET    | `/`                | Main UI                          |
| POST   | `/predict`         | Predict sarcasm in a statement   |
| GET    | `/random_quote`    | Get a random sarcastic quote     |
| GET    | `/visualizations`  | Show model performance plots     |
| POST   | `/train_model`     | Retrain the model                |

---

## 📜 License

MIT License. Free to use and modify.
