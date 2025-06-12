from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os

from utils.doc_gen import generate_doc_from_code
from utils.validator import validate_model
from utils.rag_engine import answer_query, build_kb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/build-knowledge-base")
async def build_knowledge_base(files: List[UploadFile] = File(...)):
    doc_list = []
    for file in files:
        content = await file.read()
        doc_list.append({
            "filename": file.filename,
            "content": content.decode("utf-8", errors="ignore")
        })
    result = build_kb(doc_list)
    return {"status": "Knowledge base built", "chunks": result}

class DocRequest(BaseModel):
    code: str
    model_type: str
    portfolio_type: str
    regulations: List[str]

@app.post("/generate-documentation")
def generate_documentation(request: DocRequest):
    sections = generate_doc_from_code(
        code=request.code,
        model_type=request.model_type,
        portfolio_type=request.portfolio_type,
        regulations=request.regulations
    )
    return {"status": "Documentation generated", "sections": sections}

@app.post("/run-validation")
def run_validation(request: DocRequest):
    score, issues = validate_model(request.code)
    return {
        "status": "Validation complete",
        "risk_score": score,
        "issues": issues
    }

class ChatRequest(BaseModel):
    query: str

@app.post("/chat-rag")
def chat_rag(req: ChatRequest):
    answer = answer_query(req.query)
    return {"response": answer}

@app.get("/download-doc")
def download_doc():
    file_path = "outputs/model_doc.docx"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename="model_doc.docx")
    else:
        return {"error": "File not found"}

@app.get("/download-validation-report")
def download_validation_report():
    file_path = "outputs/validation_report.docx"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename="validation_report.docx")
    else:
        return {"error": "File not found"}
