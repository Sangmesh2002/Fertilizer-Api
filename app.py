from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

# Define FastAPI app
app = FastAPI()

# Load the trained model
classifier = joblib.load("Fertclassifier-Model.pkl")

# Define input data schema using Pydantic BaseModel
class InputData(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Nitrogen: float
    Potassium: float
    Phosphorous: float
    Soil_Num: int
    Crop_Num: int

@app.post("/recommend_fertilizer/")
def recommendation(data: InputData):
    features = np.array([[data.Temparature, data.Humidity, data.Moisture, data.Nitrogen, data.Potassium, data.Phosphorous, data.Soil_Num, data.Crop_Num]])
    prediction = classifier.predict(features)
    return {"fertilizer": prediction[0]}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
