"""
app.py - FastAPI backend for AlgoRisk AI RAG-powered Risk Model Documentation/Validation

- Exposes endpoints for:
    - RAG knowledge base building
    - Model documentation generation (standard & RAG-enhanced)
    - Model validation
    - Chatbot interaction
    - File downloads (if required)
- Uses modular utils: rag_engine, doc_gen, validator
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json

# Local utils
from utils.rag_engine import SimpleRAG
from utils.doc_gen import generate_documentation_content, create_word_document, generate_rag_enhanced_documentation
from utils.validator import validate_model_documentation

# Initialize app
app = FastAPI(title="AlgoRisk AI Backend", version="1.0")

# CORS for V0 frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory RAG (use persistent store in production)
rag_system = SimpleRAG()

# ------------------------------
# 🔹 1. Build RAG Knowledge Base
# ------------------------------
@app.post("/build-knowledge-base")
def build_knowledge_base(files: List[UploadFile] = File(...)):
    documents = []
    for file in files:
        content = file.file.read().decode("utf-8", errors="ignore")
        documents.append({
            "text": content,
            "type": file.content_type,
            "source": file.filename
        })
    rag_system.build_knowledge_base(documents)
    return {"success": True, "message": f"Knowledge base built with {len(documents)} documents."}


# ------------------------------
# 🔹 2. Generate Documentation
# ------------------------------
@app.post("/generate-documentation")
def generate_documentation(
    code: UploadFile = File(...),
    model_type: str = Form(...),
    portfolio_type: str = Form(...),
    regulations: str = Form(...),  # JSON list
    results: Optional[str] = Form(None),
    data: Optional[UploadFile] = File(None),
    config: Optional[UploadFile] = File(None),
    use_rag: bool = Form(False)
):
    code_content = code.file.read().decode("utf-8", errors="ignore")
    data_content = data.file.read().decode("utf-8", errors="ignore") if data else None
    config_content = config.file.read().decode("utf-8", errors="ignore") if config else None
    results_dict = json.loads(results) if results else {}
    regulations_list = json.loads(regulations)

    # Placeholder for future code analysis
    code_analysis = {}

    # Generate documentation
    if use_rag:
        doc_content = generate_rag_enhanced_documentation(
            rag_system, code_analysis, results_dict, model_type,
            portfolio_type, regulations_list, code_content,
            data_content, config_content, additional_info="",
            workflow_placeholder=None
        )
    else:
        doc_content = generate_documentation_content(
            code_analysis, results_dict, model_type, portfolio_type,
            regulations_list, code_content, data_content, config_content
        )

    doc_buffer, filename = create_word_document(doc_content, model_type)
    return StreamingResponse(
        doc_buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ------------------------------
# 🔹 3. Run Validation
# ------------------------------
@app.post("/run-validation")
def run_validation(
    doc: UploadFile = File(...),
    code: UploadFile = File(...),
    validation_level: str = Form("Standard Review")
):
    doc_text = doc.file.read().decode("utf-8", errors="ignore")
    code_content = code.file.read().decode("utf-8", errors="ignore")
    results = validate_model_documentation(doc_text, code_content, validation_level)
    return JSONResponse(content=results)


# ------------------------------
# 🔹 4. Chat with RAG
# ------------------------------
@app.post("/chat-rag")
def chat_rag(query: str = Form(...), k: int = Form(5)):
    return rag_system.query(query, k=k)


# ------------------------------
# 🔹 5. Download Endpoints (Optional)
# ------------------------------
@app.get("/download-doc")
def download_doc():
    raise HTTPException(status_code=501, detail="Not implemented yet.")

@app.get("/download-validation-report")
def download_validation_report():
    raise HTTPException(status_code=501, detail="Not implemented yet.")


# ------------------------------
# ✅ Health Check
# ------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
