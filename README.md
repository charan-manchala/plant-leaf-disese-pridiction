# 🌿 Plant Leaf Disease Prediction System

An AI-powered web application that detects plant diseases from leaf images using deep learning and provides instant cure recommendations.

## 🔍 What It Does
- Upload any plant leaf photo
- AI identifies the plant and disease  
- Shows how much of the leaf is affected (%)
- Gives step-by-step cure and prevention tips

## 🌱 Supported Plants
Tomato | Potato | Corn | Apple | Grape | Pepper

## 🚀 Run It Yourself

**Step 1 — Clone the repo**
```bash
git clone https://github.com/charan-manchala/plant-leaf-disese-priduction.git
cd plant-leaf-disese-priduction
```

**Step 2 — Install packages**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```

## 🧠 Train the Model
```bash
python models/train_model.py \
  --dataset_path "path/to/plantvillage" \
  --plants Tomato Potato "Corn_(maize)" Apple Grape "Pepper,_bell" \
  --samples_per_class 200 \
  --epochs 10
```

## 📊 Results
| Metric | Value |
|---|---|
| Overall Accuracy | 91.2% |
| Disease Classes | 15 |
| Plants Supported | 6 |
| Model | EfficientNet-B0 |
| Training Time | ~35 mins on CPU |

## 🛠️ Technologies Used
| Tool | Purpose |
|---|---|
| PyTorch | Train and run AI model |
| EfficientNet-B0 | Disease classification |
| GradCAM | Disease heatmap visualization |
| Streamlit | Web application |
| PlantVillage | Training dataset |

## 📦 Dataset
[PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 👨‍💻 Developer
Ram Charan MANCHALA