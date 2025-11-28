from src.data_extractor import DataExtractor
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dependancies.pinecone_operations import PineconeOperations

pdf_path = "D:/georgian-chatbot/data/Labour_Code_Georgian_Version.pdf"

def upload_data(pdf_path):
    pinecone_ops = PineconeOperations()
    extractor = DataExtractor(pdf_path)
    text = extractor.extract_text()

    # Split the text into smaller chunks if necessary
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    pinecone_ops.upload_documents(documents)
    return len(documents)

if __name__ == "__main__":
    docs = upload_data(pdf_path)
    print(f"Uplaoded {docs} documents to Pinecone.")
