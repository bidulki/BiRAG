from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

documents = ["오타니 쇼헤이", "손흥민", "박지성", "크리스티아누 호날두", "켄 톰프슨", "데니스 리치",
             "크리스토퍼 콜럼버스", "니콜라 테슬라", "토마스 에디슨", "한강 (작가)", "삼성전자",
             "LG전자", "성균관대학교", "서울대학교", "다이제", "찰스 다윈", "마하트마 간디",
             "조지 워싱턴", "마틴 루터 킹", "SK 하이닉스"]

embeddings = [embedding_model.embed_query(doc) for doc in documents]

faiss_index = FAISS.from_texts(documents, embedding_model)

faiss_index.save_local("./DB/faiss_index")

print(faiss_index.similarity_search("성대", k=1)[0].page_content)