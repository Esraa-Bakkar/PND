from fastapi import FastAPI, File, UploadFile
from app.ml_service import MLService
import uvicorn
import io

app = FastAPI(title="Pneumonia Detection API")
ml_service = MLService()

@app.get("/")
def read_root():
    return {"message": "PND AI Inference API is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    #  Read the image bytes
    contents = await file.read()
    image_stream = io.BytesIO(contents)
    
    #  Get prediction from our service
    result = ml_service.predict(image_stream)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)