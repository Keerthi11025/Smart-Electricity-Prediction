# Smart Electricity Consumption Prediction and Monitoring System

## 🏆 Project Overview
A **final-year B.Tech CSE project** that uses a stacked **LSTM (Long Short-Term Memory)** deep learning model to predict household electricity consumption and deliver actionable insights via an interactive Streamlit web dashboard.

---

## 🎯 Objectives
| Objective | Status |
|-----------|--------|
| Predict next-day electricity consumption | ✅ |
| 7-day auto-regressive forecast | ✅ |
| Daily / weekly / monthly usage charts | ✅ |
| Estimated electricity bill (with tax) | ✅ |
| Statistical anomaly detection | ✅ |
| Actual vs. Predicted evaluation chart | ✅ |
| Raw data explorer + CSV download | ✅ |

---

## 🛠️ Technologies Used
| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| TensorFlow / Keras | Stacked LSTM model |
| Pandas / NumPy | Data wrangling |
| Scikit-learn | MinMaxScaler, MAE/RMSE |
| Plotly | Interactive dashboards |
| Matplotlib | Static training charts |
| Streamlit | Web application |
| Joblib | Model persistence |
| Git / GitHub | Version control |

---

## 📦 Dataset
**UCI Individual Household Electric Power Consumption**  
🔗 https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

- ~2 million minute-level readings
- 4 years of real household data
- Key column used: `Global_active_power` (kW → daily kWh)

> Download the ZIP, extract it, and place `household_power_consumption.txt` inside the `dataset/` folder.

---

## 📁 Project Structure
```
Smart_Electricity_Prediction/
│
├── dataset/
│   └── household_power_consumption.txt   ← place UCI file here
│
├── train.py                              ← preprocess + LSTM training
├── app.py                                ← Streamlit web app
├── requirements.txt                      ← Python dependencies
├── README.md
├── .gitignore
│
├── electricity_model.keras               ← generated after training
├── scaler.pkl                            ← generated after training
├── daily_usage.csv                       ← generated after training
├── training_history.json                 ← generated after training
└── evaluation_chart.png                  ← generated after training
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/Smart-Electricity-Prediction.git
cd Smart-Electricity-Prediction

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 – Train the model
```bash
python train.py
```
This will:
- Clean and resample the dataset to daily kWh
- Train a stacked LSTM model (70/15/15 split)
- Print MAE, RMSE, MAPE and a comparison against the naïve baseline
- Save `electricity_model.keras`, `scaler.pkl`, `daily_usage.csv`, `evaluation_chart.png`

### Step 2 – Launch the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 🧠 Model Architecture
```
Input  →  LSTM(64, return_sequences=True)  →  Dropout(0.2)
       →  LSTM(32)                         →  Dropout(0.2)
       →  Dense(16, relu)
       →  Dense(1)
```
- **Look-back window:** 7 days
- **Target:** Day 8 consumption (kWh)
- **Loss:** Mean Squared Error
- **Optimiser:** Adam (lr=0.001, ReduceLROnPlateau)
- **Regularisation:** EarlyStopping (patience=7) + Dropout

---

## 📊 App Features
1. **Energy Overview** – prediction, 30-day avg, estimated bill, % change
2. **Interactive Daily Chart** – rolling average overlay + anomaly markers
3. **Weekly & Monthly Breakdown** – bar charts with cost estimates
4. **Bill Estimator** – gauge chart with tax support
5. **Anomaly Detection** – configurable σ-threshold with alert band
6. **7-Day Forecast** – auto-regressive predictions table + chart
7. **Evaluation Charts** – loss curves and actual-vs-predicted on test set
8. **Raw Data Explorer** – interactive table with CSV download

---

## 📈 Performance
| Metric | Value |
|--------|-------|
| Test MAE | *see `training_history.json`* |
| Test RMSE | *see `training_history.json`* |
| Test MAPE | *see `training_history.json`* |
| Baseline comparison | LSTM vs naïve persistence |

---

## 📝 Academic Notes
- **Why LSTM?** Electricity data is sequential; LSTM captures long-range temporal dependencies better than plain RNNs or MLPs.
- **Naïve baseline comparison** is reported to prove the model adds value.
- **Data leakage prevention:** scaler is fit only on the training set.

---

## 🖼️ Screenshots
> Add screenshots of your running application here.

---

## 📄 License
MIT License – feel free to use and modify for your academic projects.

---

*Smart Electricity Consumption Prediction System | B.Tech CSE Final Year Project*
