"""
app.py - FastAPI backend for AlgoRisk AI RAG-powered Risk Model Documentation/Validation

- Exposes endpoints for:
    - RAG knowledge base building
    - Model documentation generation (standard & RAG-enhanced)
    - Model validation (standard & RAG-enhanced)
- Uses modular utils: rag_engine, doc_gen, validator
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from utils.rag_engine import SimpleRAG
from utils.doc_gen import generate_documentation_content, create_word_document, generate_rag_enhanced_documentation
from utils.validator import validate_model_documentation, signup_user, verify_login
import json

app = FastAPI(
    title="AlgoRisk AI Backend",
    description="RAG-powered Risk Model Documentation/Validation API",
    version="1.0.0"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory RAG system (for demo; use persistent store in prod) ---
rag_system = SimpleRAG()

# --- Auth Endpoints ---
@app.post("/signup")
def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if signup_user(name, email, password):
        return {"success": True, "message": "Signup successful."}
    raise HTTPException(status_code=400, detail="Email already exists.")

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    user = verify_login(email, password)
    if user:
        return {"success": True, "user": user}
    raise HTTPException(status_code=401, detail="Invalid credentials.")
from utils.gemini_quota import can_use_gemini, add_tokens_used, MONTHLY_TOKEN_LIMIT, get_tokens_used_this_month

@app.post("/doc/generate")
def generate_doc(...):
    # ...existing code...
    estimated_tokens = 1500  # Or whatever you set as max_output_tokens
    if not can_use_gemini(estimated_tokens):
        return JSONResponse(
            status_code=429,
            content={"error": f"You have exceeded your monthly Gemini token usage limit ({MONTHLY_TOKEN_LIMIT} tokens). Please try again next month."}
        )
    # ...call Gemini...
    # After successful call, record usage:
    add_tokens_used(estimated_tokens)
    # ...return response...

# --- RAG Knowledge Base Build ---
@app.post("/rag/build")
def build_knowledge_base(files: List[UploadFile] = File(...)):
    documents = []
    for file in files:
        content = file.file.read().decode("utf-8", errors="ignore")
        documents.append({"text": content, "type": file.content_type, "source": file.filename})
    rag_system.build_knowledge_base(documents)
    return {"success": True, "message": f"Knowledge base built with {len(documents)} documents."}

# --- RAG Query Endpoint ---
@app.post("/rag/query")
def rag_query(query: str = Form(...), k: int = Form(5)):
    result = rag_system.query(query, k=k)
    return result

# --- Documentation Generation ---
@app.post("/doc/generate")
def generate_doc(
    code: UploadFile = File(...),
    results: Optional[str] = Form(None),
    model_type: str = Form(...),
    portfolio_type: str = Form(...),
    regulations: str = Form(...),  # JSON list
    data: Optional[UploadFile] = File(None),
    config: Optional[UploadFile] = File(None),
    use_rag: bool = Form(False)
):
    code_content = code.file.read().decode("utf-8", errors="ignore")
    data_content = data.file.read().decode("utf-8", errors="ignore") if data else None
    config_content = config.file.read().decode("utf-8", errors="ignore") if config else None
    regulations_list = json.loads(regulations)
    results_dict = json.loads(results) if results else {}
    code_analysis = {}  # Optionally call analyze_code_structure here
    if use_rag:
        doc_content = generate_rag_enhanced_documentation(
            rag_system, code_analysis, results_dict, model_type, portfolio_type, regulations_list,
            code_content, data_content, config_content, additional_info="", workflow_placeholder=None
        )
    else:
        doc_content = generate_documentation_content(
            code_analysis, results_dict, model_type, portfolio_type, regulations_list,
            code_content_str=code_content, data_content_str=data_content, config_content_str=config_content
        )
    doc_buffer, doc_filename = create_word_document(doc_content, model_type)
    return StreamingResponse(doc_buffer, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f"attachment; filename={doc_filename}"})

# --- Model Validation ---
@app.post("/validate")
def validate_doc(
    doc: UploadFile = File(...),
    code: UploadFile = File(...),
    validation_level: str = Form("Standard Review")
):
    doc_text = doc.file.read().decode("utf-8", errors="ignore")
    code_content = code.file.read().decode("utf-8", errors="ignore")
    results = validate_model_documentation(doc_text, code_content, validation_level)
    return JSONResponse(content=results)

# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok"}
