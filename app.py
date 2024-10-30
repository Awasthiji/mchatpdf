from flask import Flask, request, jsonify, render_template, Response
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import threading
import time
import logging

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize session-based in-memory variables
pdf_data = []
vectorstore = None
qa_chain = None
status = "Waiting for PDF upload..."

# Function to initialize or update the QA chain after embeddings are created
def initialize_qa_chain():
    global qa_chain, vectorstore
    if vectorstore:
        RAG_TEMPLATE = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        <context>
        {context}
        </context>
        Answer the following question:
        {question}"""

        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        model = ChatOllama(model="llama3.2")
        retriever = vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | model
            | StrOutputParser()
        )
        logging.info("QA chain initialized or updated successfully.")

# Function to load and process PDFs, create embeddings, and update QA chain in-memory
def load_pdfs(file_paths):
    global pdf_data, vectorstore, qa_chain, status
    try:
        status = "Loading PDFs..."
        logging.info(status)
        all_texts = []

        # Load and process each PDF
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            pdf_content = loader.load()
            pdf_data.append(pdf_content)
            logging.info(f"Loaded content from {file_path}")

        # Split PDFs into chunks
        status = "Splitting PDFs into chunks..."
        logging.info(status)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        
        for document in pdf_data:
            all_texts.extend(text_splitter.split_documents(document))
        
        # Create embeddings in-memory
        status = "Creating embeddings..."
        logging.info(status)
        local_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        # Re-initialize vectorstore with new documents (in-memory)
        vectorstore = Chroma.from_documents(
            documents=all_texts, 
            embedding=local_embeddings, 
            collection_name="my_collection"  # No persist_directory specified for in-memory operation
        )

        # Update the QA chain with the new documents
        initialize_qa_chain()
        
        status = "Ready to chat with the PDFs!"
        logging.info(status)
    except Exception as e:
        status = f"Error: {str(e)}"
        logging.error(status)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdfs():
    global status
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    files = request.files.getlist("pdf")
    if len(files) > 10:
        return jsonify({"error": "You can upload a maximum of 10 PDF files."}), 400

    file_paths = []
    for file in files:
        if file and file.filename.endswith(".pdf"):
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            file_paths.append(file_path)
        else:
            return jsonify({"error": "Invalid file format. Please upload only PDF files."}), 400

    status = "PDFs uploaded. Processing..."
    logging.info(status)
    
    # Run the processing in a separate thread to avoid blocking
    threading.Thread(target=load_pdfs, args=(file_paths,)).start()
    return jsonify({"message": "PDFs uploaded and processing started."})

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": status})

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain
    question = request.json.get("question")
    logging.info(f"Received question: {question}")

    if not question:
        logging.warning("No question provided")
        return Response("Please enter a question", status=400, content_type="text/plain")
    
    if qa_chain is None:
        logging.warning("QA chain not initialized")
        return Response("Please upload a PDF first to Chat", status=400, content_type="text/plain")

    def generate():
        answer = qa_chain.invoke(question)
        for chunk in answer:
            yield chunk
            time.sleep(0.1)  # Simulate streaming delay

    return Response(generate(), content_type="text/plain")

if __name__ == "__main__":
    app.run(debug=True)
