<<<<<<< HEAD
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return ""

    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Test the function
if __name__ == "__main__":
    pdf_path = "resume_samples/resume1.pdf"  # <- removed ../ for direct path
    print(f"📂 Looking for: {pdf_path}")
    
    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text.strip() == "":
        print("⚠️ No text was extracted from the resume.")
    else:
        print("------ Extracted Resume Text ------")
        print(extracted_text)
=======
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return ""

    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Test the function
if __name__ == "__main__":
    pdf_path = "resume_samples/resume1.pdf"  # <- removed ../ for direct path
    print(f"📂 Looking for: {pdf_path}")
    
    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text.strip() == "":
        print("⚠️ No text was extracted from the resume.")
    else:
        print("------ Extracted Resume Text ------")
        print(extracted_text)
>>>>>>> 2bf27a8f92fec99ff4d05975ec9421706e1904ae
