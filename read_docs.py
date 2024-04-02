import os
from PyPDF2 import PdfFileReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_context(context, chunk_size=200, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = Document(page_content="text", metadata={"source": "local"})
    doc.page_content = context
    docs = text_splitter.split_documents([doc])
    return docs
def read_pdf_files(folder_path):
    """
    Reads all PDF files in the specified folder and returns their contents in a list.

    :param folder_path: Path to the folder containing PDF files.
    :return: List containing the contents of each PDF file.
    """
    pdf_contents = []  # List to store the contents of each PDF

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError("The specified folder does not exist.")

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)  # Full path to the file
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfFileReader(file)
                    content = ''
                    for page in range(pdf_reader.numPages):
                        content += pdf_reader.getPage(page).extractText()
                    pdf_contents.append(content)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return pdf_contents