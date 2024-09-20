from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

documents = ["오타니 쇼헤이"]

embeddings = [embedding_model.embed_query(doc) for doc in documents]

faiss_index = FAISS.from_texts(documents, embedding_model)

faiss_index.save_local("./DB/faiss_index")

print(faiss_index.similarity_search("오타니", k=1)[0].page_content)