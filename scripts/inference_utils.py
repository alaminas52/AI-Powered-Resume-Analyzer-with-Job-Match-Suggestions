import spacy

nlp = spacy.load("en_core_web_sm")

# Basic rule-based skill inference from sentence context
def infer_skills_from_text(text):
    doc = nlp(text)
    inferred = []

    # Rule examples
    if "developed" in text.lower() or "built" in text.lower():
        for token in doc:
            if token.dep_ == "dobj" or token.pos_ == "NOUN":
                inferred.append(token.text.lower())

    # Expand with noun chunks
    for chunk in doc.noun_chunks:
        if any(kw in chunk.text.lower() for kw in ["project", "system", "platform", "api"]):
            inferred.append(chunk.text.lower())

    return list(set(inferred))
