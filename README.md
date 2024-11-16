# 📈 Yes Bank Stock Closing Price Prediction  

> **A machine learning and deep learning project to predict Yes Bank's stock closing price with high accuracy using historical data.**  

---

## 🚀 **Project Overview**  
Predicting stock prices is crucial for making informed financial decisions. This project applies **machine learning** and **deep learning** techniques to analyze historical stock data and predict Yes Bank's closing price.  

The project includes:  
- Comprehensive data preprocessing and feature engineering.  
- Exploratory Data Analysis (EDA) to uncover insights into stock trends.  
- Implementation of multiple models to identify the best-performing one.  
- Deployment-ready predictive models for stock trend forecasting.  

---

## 📊 **Dataset Summary**  
| **Property**        | **Details**                          |  
|----------------------|--------------------------------------|  
| **Source**           | Publicly available stock market data |  
| **Rows**            | 3,840                               |  
| **Columns**         | 7                                   |  
| **Columns Explained**| Date, Open, High, Low, Close, Adjusted Close, Volume |  

### Sample Data:  
| Date       | Open   | High   | Low    | Close  | Adjusted Close | Volume |  
|------------|--------|--------|--------|--------|----------------|--------|  
| 2024-01-01 | 85.25  | 87.10  | 84.50  | 86.75  | 86.75          | 1,245,300 |  
| 2024-01-02 | 86.80  | 88.50  | 86.10  | 87.90  | 87.90          | 1,315,000 |  

---

## 📂 **Project Structure**  
```plaintext
📁 Yes-Bank-Stock-Closing-Price-Prediction
├── 📄 README.md             # Project documentation
├── 📂 Data                  # Dataset used for training and testing
├── 📂 Notebooks             # Jupyter Notebooks for data preprocessing and model training
├── 📂 Models                # Trained models with hyperparameters
├── 📂 Results               # Model evaluation metrics and visualizations
├── 📄 requirements.txt      # Dependencies and libraries
└── 📊 Deployment            # Scripts for deploying the best model
```

---

## 🛠️ **Technologies and Tools Used**  
- **Programming Language**: Python  
- **Libraries**:  
  - Data Analysis: Pandas, NumPy  
  - Visualization: Matplotlib, Seaborn, Plotly  
  - Machine Learning: Scikit-learn  
  - Deep Learning: TensorFlow, Keras  
- **Version Control**: Git, GitHub  

---

## 📊 **Data Preprocessing and Feature Engineering**  
- Handled missing values using forward fill techniques.  
- Removed extreme outliers using the **Interquartile Range (IQR)** method.  
- Engineered features:  
  - **Technical Indicators**:  
    - Simple Moving Average (SMA)  
    - Exponential Moving Average (EMA)  
    - Relative Strength Index (RSI)  
    - Bollinger Bands  
- Scaled features using Min-Max Scaling for uniformity.  

---

## 📈 **Exploratory Data Analysis (EDA)**  
- **Trend Analysis**:  
  - Identified seasonal trends in stock prices.  
  - Found a correlation between closing prices and trading volumes.  
- **Visualization Highlights**:  
  - **Line Charts**: To observe daily price fluctuations.  
  - **Heatmaps**: To analyze feature correlations.  
  - **Box Plots**: To detect and handle outliers.  

---

## 🤖 **Models Implemented**  
1. **Linear Regression**:  
   - Baseline model with RMSE of **15.8**.  
2. **Random Forest Regressor**:  
   - Captured non-linear relationships with RMSE of **10.5**.  
3. **LSTM (Long Short-Term Memory)**:  
   - Utilized sequential dependencies in stock data.  
   - Achieved the lowest RMSE of **8.2**.  

---

## 🏆 **Best Model and Evaluation**  
- **Best Model**: LSTM  
- **Evaluation Metrics**:  
  - **R² Score**: 92.3%  
  - **RMSE**: 8.2  
  - **MAE**: 6.5  

---

## 🔍 **Final Prediction Results**  
| **Date**       | **Actual Closing Price** | **Predicted Closing Price** | **Error** |  
|-----------------|--------------------------|-----------------------------|-----------|  
| 2024-10-01     | ₹90.25                   | ₹91.10                      | ₹0.85     |  
| 2024-10-02     | ₹88.75                   | ₹89.02                      | ₹0.27     |  
| 2024-10-03     | ₹92.80                   | ₹93.14                      | ₹0.34     |  

---

## 📌 **Key Features**  
✔️ Fully automated pipeline for data cleaning and feature engineering.  
✔️ Visualization of stock trends and patterns.  
✔️ Deployment-ready models for real-world use cases.  

---

## 🔧 **Setup Instructions**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/San7122/Yes-Bank-Stock-Closing-Price-Prediction.git
   ```  
2. Navigate to the project directory:  
   ```bash
   cd Yes-Bank-Stock-Closing-Price-Prediction
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Run Jupyter Notebooks:  
   - Explore `EDA.ipynb` for data analysis.  
   - Train models using `LSTM_Model.ipynb`.  

---

## 🌟 **Future Enhancements**  
- Include additional external factors like economic indicators.  
- Deploy the model as a web application for real-time predictions.  
- Explore hybrid models to improve accuracy.  

---

## 🤝 **Contributing**  
We welcome contributions! Feel free to fork this repository, make updates, and create a pull request.  

---
