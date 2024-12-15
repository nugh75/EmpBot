import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

# Configura il modello LLaMA locale
llama_model = ChatOpenAI(
    base_url="http://localhost:11434/v1", 
    temperature=0.5, 
    api_key="not-need", 
    model_name="llama3"
)

# Configura le embeddings con HuggingFace
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Funzione per calcolare e indicizzare le embeddings
def process_text_chunks(text):
    # Dividi il testo in chunk
    chunks = text.split('\n++++++\n')

    # Calcola le embeddings per i chunk
    chunk_embeddings = embeddings_model.embed_documents(chunks)

    # Converti le embeddings in un array numpy
    embeddings_array = np.array(chunk_embeddings)

    # Inizializza l'indice FAISS
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index, chunks

# Funzione per trovare chunk simili
def find_similar_chunks(query, index, chunks, top_k=3):
    # Calcola le embeddings per la query
    query_embedding = embeddings_model.embed_query(query)
    query_embedding_array = np.array(query_embedding).reshape(1, -1)

    # Trova i chunk più simili
    D, I = index.search(query_embedding_array, top_k)

    # Restituisci i chunk simili
    similar_chunks = [chunks[idx] for idx in I[0]]
    return similar_chunks

# Configurazione dell'interfaccia Streamlit
def main():
    st.title("Ricerca di Somiglianza nei Testi di Commozione")

    # Uploader per il file di testo
    uploaded_file = st.file_uploader("Carica un file di testo (.txt)", type="txt")

    if uploaded_file is not None:
        # Leggi il contenuto del file di testo
        text = uploaded_file.read().decode("utf-8")

        # Processa il testo e indicizza i chunk
        index, chunks = process_text_chunks(text)
        
        # Form per inserire la query
        query = st.text_input("Inserisci una query per trovare chunk simili", value="un evento che mi ha fatto piangere")

        if st.button("Cerca"):
            # Trova chunk simili
            similar_chunks = find_similar_chunks(query, index, chunks)

            # Mostra i risultati
            st.write("Chunk più simili trovati:")
            for i, chunk in enumerate(similar_chunks, start=1):
                st.write(f"**Similar Chunk {i}:**")
                st.write(chunk)
                st.write("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
