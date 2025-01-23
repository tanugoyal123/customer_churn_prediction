## CUSTOMER CHURN PREDICTION FOR TELECOM COMPANY
Customer churn prediction refers to the process of identifying customers who are likely to stop doing business with a company in the near future. This is a critical task for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. Customer churn is a critical issue faced by telecom companies, where retaining customers is often more cost-effective than acquiring new ones. 
Churn: A customer is said to have "churned" when they stop using a company's product or service.

# PROJECT OVERVIEW :
This project aims to predict customer churn for a telecom company by analyzing customer behavior and interaction patterns. The data is pre-processed and analyzed before training models using both balanced and unbalanced datasets. The final model is saved and loaded for deployment. FastAPI along with HTML templates is used for deploying the application.

# DATASET USED

The dataset used for this project contains various customer-related features such as demographics, account information, and service usage details. By analyzing these features, you can build machine learning models to predict customer churn, identify key factors leading to churn, and develop strategies to retain customers.

columns in used dataset:
Dataset Columns:
customerID: Unique ID for each customer
gender: Gender of the customer (Male/Female)
SeniorCitizen: Whether the customer is a senior citizen (1, 0)
Partner: Whether the customer has a partner (Yes/No)
Dependents: Whether the customer has dependents (Yes/No)
tenure: Number of months the customer has stayed with the company
PhoneService: Whether the customer has phone service (Yes/No)
MultipleLines: Whether the customer has multiple lines (Yes/No, No phone service)
InternetService: Customer's internet service provider (DSL, Fiber optic, No)
OnlineSecurity: Whether the customer has online security (Yes/No, No internet service)
OnlineBackup: Whether the customer has online backup (Yes/No, No internet service)
DeviceProtection: Whether the customer has device protection (Yes/No, No internet service)
TechSupport: Whether the customer has tech support (Yes/No, No internet service)
StreamingTV: Whether the customer has streaming TV (Yes/No, No internet service)
StreamingMovies: Whether the customer has streaming movies (Yes/No, No internet service)
Contract: The contract term of the customer (Month-to-month, One year, Two year)
PaperlessBilling: Whether the customer has paperless billing (Yes/No)
PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card)
MonthlyCharges: The amount charged to the customer monthly
TotalCharges: The total amount charged to the customer
Churn: Whether the customer churned (Yes/No)

The dataset mentioned above was imbalanced so two approaches are followed:

1. with balanced data 
2. with unbalanced data 

# Data Preprocessing :

1. Data Loading 
   1.1. checked null values 
   1.2. checked the data type for eah feature
   1.3. checked foe the unique categories for categorical features.

2. preprocessing:
   2.1. drop the unwanted features(customer_id)
   2.2. handling numerical feature(Total charges) woth data type is not float converted it to float
   2.3. Label encoding the categorical values.
   2.4. checked for outliers
   2.5. data splitting in training and testing [ splitted before the upsampling to evaluate on original dataset ]
   2.6. upsampling - SMOTENC(FOR BALANCED THE TRAINING DATA)(This step is performed only for balanced approach)
   2.7. Scaling of Training data 
   2.8. correlation analysis
   2.9. PCA for multicollinearity (COrrelation was there)
   2.10. saving test and traing dataset
   [For unbalanced approach upsampling step was not performed]

# Model and Training :
   Model choosed for classification :
   1. Logistic regression
   2. K-Nearest Neighbour[only for balanced]
   3. Decision Tree
   4. Randomn Forest
   5. SVM 
   6. Naive bayes[only for balanced]

   KNN and Naive bayes works well for balanced data and other models handles unbalanced data very well so KNN and Naive bayes are nor used for unbalanced data.

# Evaluation :
  The models were evaluated using different metrics follows:
  1. Training accuracy
  2. Testing accuracy
  3. cross-val score
  4. precision 
  5. Recall 

# Results :
Results for each model are as follows:

1. For balanced data 

| Model               | Training Accuracy | Testing Accuracy |CV_Score   | Precision | Recall | F1-Score |
|---------------------|-------------------|------------------|-----------|-----------|--------|----------|
| Logistic Regression | 80.5%             | 74.0%            | 79.6%     | 70%       | 74%    | 70%      |
| KNN                 | 86.3%             | 72.6%            | 80.5%     | 68%       | 72%    | 69%      |
| Decision Tree       | 99.9%             | 70.6%            | 78.8%     | 64%       | 65%    | 64%      |
| Random Forest       | 99.9%             | 75.3%            | 83.1%     | 69%       | 70%    | 69%      |
| SVM                 | 79.9%             | 74.1%            | 79.2%     | 69%       | 73%    | 70%      |
| Naive Bayes         | 79.4%             | 73.4%            | 78.9%     | 69%       | 72%    | 69%      |


2. For unbalanced data 

| Model               | Training Accuracy | Testing Accuracy |CV_Score   | Precision | Recall | F1-Score |
|---------------------|-------------------|------------------|-----------|-----------|--------|----------|
| Logistic Regression | 80.2%             | 79.4%            | 80.8%     | 74%       | 72%    | 73%      |
| Decision Tree       | 99.8%             | 71.7%%           | 73.6%     | 64%       | 65%    | 64%      |
| Random Forest       | 99.8%             | 77.4%            | 79.1%     | 71%%      | 68%    | 69%      |
| SVM                 | 80.1%             | 79.8%            | 79.8%     | 74%       | 72%    | 73%      |

The above metrics are used to compare diffrent model and it was found that Linear Regression model with unbalanced training data gives the best results and do not overfit also, decision tree and Randomn forest gave good results but overfitting.

# Deployment :

This section provides instructions to run and deploy the application built with FastAPI and HTML templates.

 Pre-Requisite :
 1. Python: Ensure Python 3.10 or later is installed. 
 2. Dependencies: Install the required Python packages listed in requirements.txt.

**Steps to Deploy Locally**

1. Clone the Repository
```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```
2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the Application locally
```bash
uvicorn app.main:app --reload
```
5. Access the Application
Open your web browser and go to:
```bash
http://127.0.0.1:8000
```










