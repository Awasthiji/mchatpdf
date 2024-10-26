from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import threading

app = Flask(__name__)

# Initialize variables
pdf_data = None
vectorstore = None
qa_chain = None
status = "Waiting for PDF upload..."

# Load PDF and set up the RAG chain
def load_pdf(file_path):
    global pdf_data, vectorstore, qa_chain, status
    try:
        status = "Loading PDF..."
        loader = PyPDFLoader(file_path)
        pdf_data = loader.load()

        status = "Splitting PDF into chunks..."
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(pdf_data)
        
        status = "Creating embeddings..."
        local_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
        
        status = "Setting up QA chain..."
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

        status = "Ready to chat with the PDF!"
    except Exception as e:
        status = f"Error: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global status
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file = request.files.get("pdf")
    if file and file.filename.endswith(".pdf"):
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        status = "PDF uploaded. Processing..."
        
        # Run the processing in a separate thread to allow status checking
        threading.Thread(target=load_pdf, args=(file_path,)).start()
        return jsonify({"message": "PDF uploaded and processing started."})
    
    return jsonify({"error": "Invalid file format. Please upload a PDF file."}), 400

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": status})

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if qa_chain is None:
        return jsonify({"error": "Please upload a PDF first"}), 400

    answer = qa_chain.invoke(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
