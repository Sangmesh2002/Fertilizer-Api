from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the classifier model from the pickle file
with open("Fertclassifier-Model.pkl", "rb") as f:
    classifier = pickle.load(f)

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
    features = [[data.Temparature, data.Humidity, data.Moisture, data.Nitrogen, data.Potassium, data.Phosphorous, data.Soil_Num, data.Crop_Num]]
    prediction = classifier.predict(features)
    return {"fertilizer": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
