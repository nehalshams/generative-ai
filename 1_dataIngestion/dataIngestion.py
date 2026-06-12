from langchain_core.documents import Document
from langchain_docling import DoclingLoader

#----------------------------- load txt file

file_path = "1_dataIngestion/speech.txt"
with open(file_path, 'r') as file:
    text = file.read()

# create document object
document = Document(
    page_content= text, 
    metadata= {"source": file_path, "file_type": "txt"}
    )

print(document)
print(f"Document: {document.page_content}")



# -----------------------------

loader = DoclingLoader(file_path=file_path)

# Load all documents
documents = loader.load()
print(f"Documents: {documents}")


# ----------------------------- Reading a pdf file


