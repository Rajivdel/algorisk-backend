import os
from docx import Document
from typing import List, Dict

def create_docx_documentation(sections: List[Dict[str, str]], save_path: str) -> str:
    """
    Generate a Word document from AI-generated documentation sections.

    Args:
        sections: List of sections with title and content
        save_path: File path where the document will be saved

    Returns:
        The path to the saved Word document
    """
    doc = Document()
    doc.add_heading("Model Documentation", 0)

    for section in sections:
        doc.add_heading(section["title"], level=1)
        doc.add_paragraph(section["content"])

    doc.save(save_path)
    return save_path


def simulate_ai_documentation(code_text: str, user_comments: str = "") -> List[Dict[str, str]]:
    """
    Simulates AI-generated model documentation from uploaded code and comments.

    Args:
        code_text: Source code content uploaded by user
        user_comments: Optional text area input from user

    Returns:
        A list of sections (each with title and content)
    """
    doc_sections = [
        "Executive Summary",
        "Model Overview",
        "Data Description",
        "Methodology",
        "Model Development",
        "Model Validation",
        "Performance Metrics",
        "Limitations",
        "Regulatory Compliance",
        "Risk Assessment",
        "Implementation Guidelines",
        "Monitoring Framework",
        "Governance Structure",
        "Documentation Standards",
        "Change Management",
        "Appendices",
        "References",
        "Glossary",
    ]

    # Simulated content generation per section
    generated = []
    for title in doc_sections:
        content = f"""
        This section titled '{title}' has been auto-generated using code analysis and uploaded configurations.
        - Code Insight: {code_text[:200]}...
        - Comments Included: {user_comments[:200]}...
        - This content follows regulatory best practices and model governance standards.
        """
        generated.append({"title": title, "content": content.strip()})

    return generated
