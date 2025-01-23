from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
from fastapi.templating import Jinja2Templates
from fastapi import Request


app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
@app.get("/")
async def form_page(request:Request):
    return templates.TemplateResponse("form.html", {"request": request})

with open('unbalanced/scaling_model.pkl', 'rb') as f:
    scaling_model = pickle.load(f)
with open('unbalanced/pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)
with open('unbalanced/final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

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
    customer_df = pd.DataFrame([customer_data])
    categorical_columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod','SeniorCitizen']
    numerical_columns=["tenure","MonthlyCharges","TotalCharges"]
    X=scaling_model.transform(customer_df[numerical_columns])
    X = pd.DataFrame(X,columns=numerical_columns)
    customer_df[numerical_columns] = X
    pc_df=pca_model.transform(customer_df[numerical_columns])

    pc_df = pd.DataFrame(data=pc_df, columns=['PC1', 'PC2'])

    d=customer_df[categorical_columns]
    X_final = pd.concat([d, pc_df], axis=1)
    pred=final_model.predict(X_final)
    if pred[0]==0:
        preds="customer is not churned"
    elif pred[0]==1:
        preds="customer is churned"
    else:
        preds="not predicted"

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