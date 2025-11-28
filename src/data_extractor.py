from langchain_community.document_loaders import PyPDFLoader

class DataExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        loader = PyPDFLoader(self.pdf_path)

        # Load and parse the PDF
        docs = loader.load()
        
        # Initialize an empty string to hold the extracted text
        full_text = ""
        
        # Iterate through each page in the PDF
        for page_num in range(len(docs)):
            text = docs[page_num].page_content               # Extract text from the page
            full_text += text + "\n"             # Append the text to full_text with a newline
        
        return full_text
