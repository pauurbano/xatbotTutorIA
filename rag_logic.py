import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import time
from datetime import datetime
from langchain.callbacks import get_openai_callback


RETRIEVER_LOG = "retriever_log.txt"

# Cargar variables de entorno
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# === Configuració ===
CHROMA_DIR = "chroma_db"
HASH_FILE = "pdf_hash.txt" 
PDF_PATH = "NORMES.pdf"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# === Funciones auxiliares ===

def calcular_hash_pdf(pdf_path):
    """Calcula el hash del archivo PDF"""
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def hash_guardado():
    """Lee el hash guardado si existe"""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, 'r') as f:
            return f.read().strip()
    return None

def guardar_hash(hash_value):
    """Guarda el hash actual"""
    with open(HASH_FILE, 'w') as f:
        f.write(hash_value)

def generar_chroma():
    """Crea una nueva base de datos Chroma"""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len
    )
    docs_split = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        docs_split,
        embedding,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    return vectordb

# === Lógica de carga/generación ===

# Calcular hash actual del PDF
hash_actual = calcular_hash_pdf(PDF_PATH)
hash_anterior = hash_guardado()

if (not os.path.exists(CHROMA_DIR)) or (hash_actual != hash_anterior):
    # Si no hay base de datos o el PDF ha cambiado, generamos de nuevo
    print("Generando nueva base de datos Chroma...")
    vectordb = generar_chroma()
    guardar_hash(hash_actual)
else:
    # Si todo está igual, cargamos la base de datos
    print("Cargando base de datos Chroma existente...")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# Crear el retriever y la cadena de QA
retriever = vectordb.as_retriever(search_type="similarity",
    search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
    retriever=retriever
)

# === Función principal para Flask ===
def get_response(user_input):
    start_time = time.time()

    with get_openai_callback() as cb:
        result = qa.invoke({"query": user_input})

    end_time = time.time()
    duration = end_time - start_time

    # Guardar en el log
    with open(RETRIEVER_LOG, 'a', encoding='utf-8') as log_file:
        log_file.write(
            f"[{datetime.now()}] Temps: {duration:.2f}s | Prompt tokens: {cb.prompt_tokens} | "
            f"Completion tokens: {cb.completion_tokens} | Total tokens: {cb.total_tokens} | "
            f"Cost: ${cb.total_cost:.6f} | Consulta: {user_input}\n"
        )

    return result['result']