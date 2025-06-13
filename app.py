from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile

app = FastAPI()

# Enable CORS for local and v0 frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to your v0 frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running", "message": "Welcome to AlgoRisk Backend API"}

@app.post("/build-knowledge-base")
async def build_knowledge_base(files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        with open(os.path.join(tempfile.gettempdir(), file.filename), 'wb') as f:
            f.write(contents)
    return {"status": "success", "message": "Knowledge base built from uploaded files"}

@app.post("/generate-documentation")
async def generate_documentation(comments: str = Form(...)):
    dummy_doc_path = os.path.join(tempfile.gettempdir(), "model_documentation.docx")
    with open(dummy_doc_path, "w") as f:
        f.write("AI-Generated Documentation\n\n" + comments)
    return {"status": "success", "message": "Documentation generated", "download_url": "/download-doc"}

@app.post("/run-validation")
async def run_validation(level: str = Form(...)):
    return {
        "status": "success",
        "validation_level": level,
        "results": {
            "overall_score": 85,
            "compliant": 12,
            "non_compliant": 3,
            "warnings": 2
        }
    }

@app.post("/chat-rag")
async def chat_rag(prompt: str = Form(...)):
    return {"response": f"Simulated AI answer for: {prompt}"}

@app.get("/download-doc")
def download_doc():
    dummy_path = os.path.join(tempfile.gettempdir(), "model_documentation.docx")
    if not os.path.exists(dummy_path):
        return JSONResponse(status_code=404, content={"detail": "Documentation not found"})
    return FileResponse(dummy_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename="model_documentation.docx")

@app.get("/download-validation-report")
def download_validation_report():
    dummy_report = os.path.join(tempfile.gettempdir(), "validation_report.txt")
    with open(dummy_report, "w") as f:
        f.write("Validation report: Overall Score = 85%, 3 issues found.")
    return FileResponse(dummy_report, media_type="text/plain", filename="validation_report.txt")
