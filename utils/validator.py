def validate_model(code):
    if "def" in code:
        return "B", ["No critical errors found."]
    else:
        return "C", ["Missing function definitions."]
