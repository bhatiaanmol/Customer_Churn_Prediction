# Customer Churn Prediction

An end-to-end machine learning project to predict customer churn and optimize recall using business-driven evaluation and threshold tuning.

---

## 📌 Problem Statement
Customer churn is a major challenge for subscription-based businesses.  
The objective of this project is to predict whether a customer is likely to churn and **prioritize recall** to minimize customer loss.

---

## 🧠 Key Insights from EDA
- Customers with **short tenure** churn significantly more
- **Higher monthly charges** are associated with higher churn
- **Month-to-month contracts** have the highest churn rate
- Lack of **tech support / online security** increases churn
- Dataset is **imbalanced (~26% churn)** → recall is more important than accuracy

---

## 🛠️ Feature Engineering
- Converted `TotalCharges` to numeric and handled missing values
- Dropped non-informative `customerID`
- One-hot encoded categorical features
- Removed multicollinearity caused by redundant service features
- Scaled numerical features using `StandardScaler`
- Used stratified train-test split to preserve churn distribution

---

## 🤖 Modeling Approach

### Logistic Regression (Baseline)
- Used as an interpretable baseline model
- Performed threshold tuning to optimize recall

### Threshold Tuning
Instead of using the default 0.5 threshold, multiple thresholds were tested:

| Threshold | Churn Recall |
|----------|-------------|
| 0.5 | ~55% |
| **0.4 (Final)** | **~67%** |
| 0.3 | ~74% (too many false positives) |

Final threshold chosen: **0.4**

---

## 🌳 Model Comparison
A Random Forest model was trained to capture non-linear interactions.

**Result:**
- Random Forest achieved higher accuracy
- Logistic Regression achieved **better recall for churn**
- Logistic Regression chosen as final model due to business alignment and interpretability

---

## 📈 Final Conclusion
This project demonstrates that model success depends on **business objectives**, not just accuracy.  
By tuning the decision threshold and prioritizing recall, a simpler Logistic Regression model outperformed a more complex model for churn prediction.

---

## 🧰 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🚀 Future Improvements
- Cost-sensitive learning
- Gradient Boosting (XGBoost / LightGBM)
- Deployment with Streamlit or FastAPI

---

## 📁 Project Structure
Customer_Churn_Prediction/
├── Data/
├── Notebooks/
├── README.md
├── requirements.txt
└── .gitignore