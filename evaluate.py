from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from agent_valid import BiRAGAgent
import pandas as pd
import json
import shutil
import os

def load_explorer(faiss_index_path, embedding_model):
    explorer = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
    return explorer

with open("valid.json", 'r', encoding="utf-8") as f:
    valid_dataset = json.load(f)

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
explorer = load_explorer("./DB/faiss_index", embedding_model)

document_dict = {
    "오타니 쇼헤이": "docs_0.json",
    "손흥민": "docs_1.json",
    "박지성": "docs_2.json",
    "크리스티아누 호날두": "docs_3.json",
    "켄 톰프슨": "docs_4.json",
    "데니스 리치": "docs_5.json",
    "크리스토퍼 콜럼버스": "docs_6.json",
    "니콜라 테슬라": "docs_7.json",
    "토마스 에디슨": "docs_8.json",
    "한강 (작가)": "docs_9.json",
    "삼성전자": "docs_10.json",
    "LG전자": "docs_11.json",
    "성균관대학교": "docs_12.json",
    "서울대학교": "docs_13.json",
    "다이제": "docs_14.json",
    "찰스 다윈": "docs_15.json",
    "마하트마 간디": "docs_16.json",
    "조지 워싱턴": "docs_17.json",
    "마틴 루터 킹": "docs_18.json",
    "SK 하이닉스": "docs_19.json"
}

for key in document_dict.keys():
    document = document_dict[key]
    document_path = os.path.join("./DB", document)
    original_path = os.path.join("./DB/faiss_index", document)
    os.remove(document_path)
    shutil.copyfile(original_path, document_path)

first_answer_list = []
answer_list = []
predict_list = []

df = pd.DataFrame()

for data in valid_dataset['test']:
    document = data['document']
    question = data['question']
    request = data['request']
    answer = data['answer']
    agent = BiRAGAgent(explorer, document_dict, document)
    print(agent.info)
    # try:
    #     first_answer = agent(question)
    #     print(f"first_answer: {first_answer}")
    #     print(agent(request))
    #     agent.reset_history()
    #     predict = agent(question)
    # except:
    #     first_answer = "halted"
    #     predict = "halted"
    # first_answer_list.append(first_answer)
    # answer_list.append(answer)
    # predict_list.append(predict)
    # print(f"predict: {predict}")
    # print(f"answer: {answer}")
    # if answer.lower() == predict.lower():
    #     print("정답")
    # else:
    #     print("오답")
    
    # for key in document_dict.keys():
    #     document = document_dict[key]
    #     document_path = os.path.join("./DB", document)
    #     original_path = os.path.join("./DB/faiss_index", document)
    #     os.remove(document_path)
    #     shutil.copyfile(original_path, document_path)
    break

# df['predict'] = predict_list
# df['answer'] = answer_list
# df['first_answer'] = first_answer_list

# df.to_csv("result_r_5_mini.tsv", sep="\t", index=False)