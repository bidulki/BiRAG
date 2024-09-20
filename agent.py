from pydantic import BaseModel
from openai import OpenAI
from typing import Literal
from search import EmbeddingFaiss
from prompt import ACTION_PROMPT, SEARCH_PROMPT, EDIT_PROMPT, NORMAL_PROMPT
import json

title2path = {
    "오타니 쇼헤이": "./DB/docs_0.json"
}

class ActionResponse(BaseModel):
    action: Literal["reset", "normal", "search", "edit",]

class SearchResponse(BaseModel):
    target: str

class EditResponse(BaseModel):
    action: Literal["add", "delete", "change"]
    subtitle_1: str
    subtitle_2: str
    idx: int
    changes: str

class BiRAGAgent():
    def __init__(self, explorer, document_dict):
        self.client = OpenAI()
        self.model = "gpt-4o-2024-08-06"
        self.search_engine = EmbeddingFaiss(explorer, document_dict, "./DB")
        self.history = []
        self.data = None
        self.search_target = "오타니 쇼헤이"
        self.info = None

    def make_message(self, role, content):
        return {"role": role, "content": content}
    
    def gpt_agent_pydantic(self, messages, format):
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=format)
        return response.choices[0].message.parsed
    
    def gpt_agent(self):
        system_prompt = self.make_message("system", NORMAL_PROMPT)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_prompt] + self.history,
        )
        return response.choices[0].message.content.strip()
    
    def gpt_agent_stream(self):
        system_prompt = self.make_message("system", NORMAL_PROMPT)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[system_prompt] + self.history,
            stream=True
        )
        return stream
        
    def reset_history(self):
        self.history = []
        return
    
    def action_selector(self, user_input):
        user_message = self.make_message("user", user_input)
        self.history.append(user_message)
        action_messages = [self.make_message("system", ACTION_PROMPT)]
        for history in self.history:
            action_messages.append(history)
        action_type = self.gpt_agent_pydantic(action_messages, ActionResponse).action
        return action_type

    def search_document(self):
        search_messages = [self.make_message("system", SEARCH_PROMPT)]
        for history in self.history:
            search_messages.append(history)
        search_target = self.gpt_agent_pydantic(search_messages, SearchResponse).target
        info, data = self.search_engine(search_target)
        return search_target, data, info
    
    def answer_Q(self, search_target, data):
        self.history.append(self.make_message("assistant", f"{search_target} 검색 완료\n{str(data)}\n답변:"))
        stream = self.gpt_agent_stream()
        return stream

    def save_json(self):
        file_path = title2path[self.search_target]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent="\t", ensure_ascii=False)

    def edit_document(self):
        edit_messages = [self.make_message("system", EDIT_PROMPT)]
        for history in self.history:
            edit_messages.append(history)
        edit_messages.append(self.make_message("user", f"검색된 문서의 구조는 다음과 같다. {self.info} 예를 들면 subtitle_1이 '인물'이고 subtitle_2가 '기본 정보'가 될 수 있다."))
        edit_response = self.gpt_agent_pydantic(edit_messages, EditResponse)
        print(edit_response)
        if edit_response.action == "add":
            response =  self.add_document(edit_response)
        elif edit_response.action == "delete":
            response = self.delete_document(edit_response)
        elif edit_response.action == "change":
            response = self.change_document(edit_response)
        self.save_json()
        return response
    
    def add_document(self, edit_response):
        subtitle_1 = edit_response.subtitle_1
        subtitle_2 = edit_response.subtitle_2
        changes = edit_response.changes
        idx = edit_response.idx
        if subtitle_2!= None:
            self.data[subtitle_1][subtitle_2].insert(idx, changes)
        else:
            self.data[subtitle_1].insert(idx, changes)
        return "내용을 추가했습니다."
    
    def change_document(self, edit_response):
        subtitle_1 = edit_response.subtitle_1
        subtitle_2 = edit_response.subtitle_2
        changes = edit_response.changes
        idx = edit_response.idx
        if subtitle_2!= None:
            self.data[subtitle_1][subtitle_2][idx] = changes
        else:
            self.data[subtitle_1][idx] = changes
        return "내용을 변경했습니다."
    
    def delete_document(self, edit_response):
        subtitle_1 = edit_response.subtitle_1
        subtitle_2 = edit_response.subtitle_2
        idx = edit_response.idx
        if subtitle_2!= None:
            del self.data[subtitle_1][subtitle_2][idx]
        else:
            del self.data[subtitle_1][idx]
        return "내용을 삭제했습니다."

    def __call__(self, user_input):
        action_type = self.action_selector(user_input)
        print(action_type)
        if action_type == "reset":
            self.reset_history()
            return "대화 기록이 삭제되었습니다."
        elif action_type == "normal":
            stream = self.gpt_agent_stream()
            return stream
        elif action_type == "search":
            search_target, data, info = self.search_document()
            self.data = data
            self.info = info
            stream = self.answer_Q(search_target, data)
            return stream
        elif action_type == "edit":
            return self.edit_document()

        