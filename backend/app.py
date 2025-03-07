from fastapi import FastAPI, File, UploadFile
import pdfplumber
import pytesseract
from PIL import Image
import io

# Explicitly set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI is running successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()

    # Extract text from PDF
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    
    # Extract text from image
    elif file.filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)

    else:
        return {"error": "Unsupported file format"}

    # Return extracted text in JSON format
    return {"filename": file.filename, "extracted_text": text}
