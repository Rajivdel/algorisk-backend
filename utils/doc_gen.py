def generate_doc_from_code(code, model_type, portfolio_type, regulations):
    return {
        "Executive Summary": f"Generated doc for {model_type} with regulations: {', '.join(regulations)}",
        "Code Summary": f"{len(code.splitlines())} lines of code."
    }
