from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
from fastapi.templating import Jinja2Templates
from fastapi import Request

#initialise fastapi App
app = FastAPI()
# Setting up Jinja2 templates for rendering HTML pages
templates = Jinja2Templates(directory="templates")

# Defining the data model for customer input using Pydantic
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# Route to display the form page
@app.get("/")
async def form_page(request:Request):
    # Rendering the form.html template
    return templates.TemplateResponse("form.html", {"request": request})
# Loading the scaling models 
with open('unbalanced/scaling_model.pkl', 'rb') as f:
    scaling_model = pickle.load(f)
#loading the pca model
with open('unbalanced/pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)

#loading the final model
with open('unbalanced/final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# defining endpoint for prediction
@app.post("/submit")
async def submit_form( request: Request,gender: int = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: int = Form(...),
    Dependents: int = Form(...),
    tenure: float = Form(...),
    PhoneService: int = Form(...),
    MultipleLines: int = Form(...),
    InternetService: int = Form(...),
    OnlineSecurity: int = Form(...),
    OnlineBackup: int = Form(...),
    DeviceProtection: int = Form(...),
    TechSupport: int = Form(...),
    StreamingTV: int = Form(...),
    StreamingMovies: int = Form(...),
    Contract: int = Form(...),
    PaperlessBilling: int = Form(...),
    PaymentMethod: int = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)):

 # Creating a dictionary to store customer data from form submission
    customer_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    # Converting dictionary to DataFrame for processing
    customer_df = pd.DataFrame([customer_data])
    # Defining categorical and numerical columns
    categorical_columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod','SeniorCitizen']
    numerical_columns=["tenure","MonthlyCharges","TotalCharges"]
    # performing scaling
    X=scaling_model.transform(customer_df[numerical_columns])
    X = pd.DataFrame(X,columns=numerical_columns)
    customer_df[numerical_columns] = X
    # performiing pca
    pc_df=pca_model.transform(customer_df[numerical_columns])

    pc_df = pd.DataFrame(data=pc_df, columns=['PC1', 'PC2'])

    d=customer_df[categorical_columns]
    X_final = pd.concat([d, pc_df], axis=1)
    # prediction using trained model
    pred=final_model.predict(X_final)
    if pred[0]==0:
        preds="customer is not churned"
    elif pred[0]==1:
        preds="customer is churned"
    else:
        preds="not predicted"
    # Returning the result to the frontend using the form template
    return templates.TemplateResponse("form.html", {"request": request,"gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,"churn_prediction": preds})