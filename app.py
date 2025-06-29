import os
from flask import Flask, request, render_template
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-key-here"

app = Flask(__name__)
UPLOAD_FOLDER = 'pdfs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_and_vectorize(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(pages, embeddings)
    return vectordb

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        file = request.files["pdf"]
        question = request.form["question"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        db = load_and_vectorize(path)
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        answer = qa_chain.run(question)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
