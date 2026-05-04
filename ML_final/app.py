import gradio as gr
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age):

    input_data = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Diabetes Detected (Outcome = 1)"
    else:
        return "No Diabetes (Outcome = 0)"


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction Web App",
    description="This app predicts whether a patient has diabetes using a trained Machine Learning model."
)

demo.launch(share=True)