import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
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

# Percorso del file di commozione
file_path = '/home/nugh75/git-repository/chatbotc/commozione.txt'

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

# Configurazione del prompt
prompt_template = ChatPromptTemplate.from_template(
    "Sei un detector di commozioni. Valuta se il seguente episodio è un caso di commozione o no: \n\nEpisodio: {question} \n\nContesto simile: {context} \n\nFornisci una risposta con 'Sì' se è un caso di commozione, altrimenti rispondi 'No'."
)

# Funzione per analizzare il sentiment dell'episodio utilizzando il modello e il prompt
def sentiment_analysis_with_context(llama_model, episode, similar_chunks):
    # Prepara il contesto dai chunk simili
    context = "\n".join(similar_chunks)

    # Creare il prompt con il contesto e l'episodio
    prompt = prompt_template.format(question=episode, context=context)
    
    # Esegui il modello LLaMA
    response = llama_model(prompt)

    # Estrai il testo dalla risposta del modello
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Analizza la risposta del modello per determinare il sentiment
    return response_text.strip()

# Configurazione dell'interfaccia Streamlit
def main():
    st.title("Analisi di Commozione degli Episodi")

    # Leggi il contenuto del file di commozione
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Processa il testo e indicizza i chunk
    index, chunks = process_text_chunks(text)

    # Form per inserire un episodio
    episode = st.text_area("Racconta un episodio di commozione")

    if st.button("Analizza"):
        # Trova chunk simili
        similar_chunks = find_similar_chunks(episode, index, chunks)

        # Mostra i chunk più simili
        st.write("Episodi di commozione più simili trovati:")
        for i, chunk in enumerate(similar_chunks, start=1):
            st.write(f"**Similar Chunk {i}:**")
            st.write(chunk)
            st.write("\n" + "-"*50 + "\n")

        # Analizza il sentimento con contesto
        sentiment = sentiment_analysis_with_context(llama_model, episode, similar_chunks)

        # Mostra il risultato dell'analisi del sentimento
        st.write(f"Risultato dell'analisi del sentimento: **{sentiment}**")

if __name__ == "__main__":
    main()
