from pydantic import BaseModel
from openai import OpenAI
from typing import Literal
from search import EmbeddingFaiss
from prompt import ACTION_PROMPT, SEARCH_PROMPT, NORMAL_PROMPT
import copy
import json
import utils


EDIT_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 내용을 추가하거나 삭제하거나 또는 수정해야한다.
검색된 문서는 json형태로 중첩된 딕셔너리 속에 문장들의 리스트로 구성되어있다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
현재 너의 문장안에서의 위치는 {current_path}이다. 사용자의 요구에 응하기 위해 삭제/추가/수정해야 하는
문서의 위치로 가기위한 다음 path를 출력하시오.

당신이 사용할 수 있는 Action 목록과 설명은 아래와 같다.
add: 사용자가 말한 정보가 기존 문서에 없을 때 문서에 문장을 추가하는 Action이다.
delete: 사용자가 기존 문서에 있는 문장을 완전히 삭제해야 할때 사용하는 Action이다.
change: 사용자가 말한 정보가 기존 문서에 있는 내용과 달라서, 문장을 수정할때 혹은 문장의 일부만 삭제할때 사용하는 Action이다.
"""

ADD_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 문서에 문장을 추가해야 한다.
검색된 문서는 json형태로 새로운 문장을 추가할 위치는 {current_path}의 리스트의 idx이다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
새로 추가할 내용을 add_text에 넣고, 그 위치를 idx로 지정해라.
"""

DELETE_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 문서에서 문장을 삭제해야 한다.
검색된 문서는 json형태로 삭제할 문장의 위치는 {current_path}의 리스트의 idx 문장이다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
삭제할 문장의 위치를 idx로 지정해라.
"""

CHANGE_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 문서에 문장을 수정해야 한다.
검색된 문서는 json형태로 문장을 수정할 위치는 {current_path}의 리스트의 idx이다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
수정이 완료된 문장을 change_text에 넣고, 그 위치를 idx로 지정해라.
"""

with open("./document.json", 'r') as f:
    document_json = json.load(f)

title2path = {}
for document in document_json.keys():
    title2path[document] = document_json[document]['file_path']    


class ActionResponse(BaseModel):
    action: Literal["reset", "normal", "search", "edit",]

class SearchResponse(BaseModel):
    target: str

class changeResponse(BaseModel):
    idx: int
    change_text: str

class addResponse(BaseModel):
    idx: int
    add_text: str

class deleteResponse(BaseModel):
    idx: int

class BiRAGAgent():
    def __init__(self, explorer, document_dict):
        self.client = OpenAI()
        self.model = "gpt-4o-2024-08-06"
        self.search_engine = EmbeddingFaiss(explorer, document_dict, "./DB")
        self.history = []
        self.data = None
        self.search_target = ""
        self.info = None

    def make_message(self, role, content):
        return {"role": role, "content": content}
    
    def gpt_agent_pydantic(self, messages, format):
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=format)
        return response.choices[0].message.parsed
    
    def gpt_agent_json(self, messages, format):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=format)
        
        return response.choices[0].message.content
    
    def create_pathfinder_schema(self, subtitle_candidates):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "pathfinder",
                "schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "현재 위치에서 문서에서 수정/삭제/추가 해야하는 위치에 도달하기 위한 다음 경로",
                            "enum": list(subtitle_candidates)  # Convert to list here
                        },
                        "action": {
                            "type": "string",
                            "description": "문서에서 수정/삭제/추가 중 무엇을 할 지 정하는 변수",
                            "enum": ["add", "change", "delete"]
                        }
                    },
                    "required": ["path", "action"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        return response_format

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
        document, info, data = self.search_engine(search_target)
        self.search_target = document
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
        edit_messages.append(self.make_message("user", f"검색된 문서의 구조는 다음과 같다. {self.info}"))

        edit_path = []
        target = self.data
        while True:
            if type(target) == list:
                break
            subtitle_candidates = target.keys()
            path_messages = copy.deepcopy(edit_messages)
            current_path = "/".join(edit_path)
            path_messages.append(self.make_message("user", EDIT_PROMPT.format(current_path=current_path)))
            path_response = self.gpt_agent_json(path_messages, self.create_pathfinder_schema(subtitle_candidates))
            print(path_response)
            path_response = json.loads(path_response)
            edit_path.append(path_response["path"])
            target = target[path_response["path"]]

        current_path = "/".join(edit_path)
        edit_messages.append(self.make_message("user", f"너의 현재 위치는 {current_path}, 수행해야할 액션은 {path_response['action']}이다."))
        
        if path_response["action"] == "add":
            edit_messages.append(self.make_message("user", ADD_PROMPT.format(current_path=current_path)))
            add_response = self.gpt_agent_pydantic(edit_messages, addResponse)
            print(add_response)
            response =  self.add_document(edit_path, add_response)

        elif path_response["action"] == "delete":
            edit_messages.append(self.make_message("user", DELETE_PROMPT.format(current_path=current_path)))
            delete_response = self.gpt_agent_pydantic(edit_messages, deleteResponse)
            print(delete_response)
            response = self.delete_document(edit_path, delete_response)

        elif path_response["action"] == "change":
            edit_messages.append(self.make_message("user", CHANGE_PROMPT.format(current_path=current_path)))
            change_response = self.gpt_agent_pydantic(edit_messages, changeResponse)
            print(change_response)
            response = self.change_document(edit_path, change_response)
        
        self.save_json()
        return response
    
    def add_document(self, edit_path, add_response):
        path_list = edit_path
        add_text = add_response.add_text
        idx = add_response.idx
        target = self.data
        for path in path_list:
            target = target[path]
        if idx >= len(target):
            target.append(add_text)
        else:
            target.insert(idx, add_text)

        return "내용을 추가했습니다."
    
    def change_document(self, edit_path, change_response):
        path_list = edit_path
        idx = change_response.idx
        change_text = change_response.change_text
        target = self.data
        for path in path_list:
            target = target[path]
        target[idx] = change_text

        return "내용을 변경했습니다."
    
    def delete_document(self, edit_path, delete_response):
        path_list = edit_path
        idx = delete_response.idx
        target = self.data
        for path in path_list:
            target = target[path]
        del target[idx]

        return "내용을 삭제했습니다."

    def __call__(self, user_input):
        action_type = self.action_selector(user_input)
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
            utils.print_log(f"search_target: {self.search_target}")
            stream = self.answer_Q(search_target, data)
            return stream
        elif action_type == "edit":
            return self.edit_document()

        