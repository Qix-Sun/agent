# File: cot_ophthalmology_agent.py
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.manager import CallbackManager
from tools.image_classifier_tool import ImageClassifierTool
from tools.ultrasound_report_tool import ultrasound_report_tool
from tools.fundus_report_tool import fundus_report_tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from RAG import rag_chain
from langchain_openai import ChatOpenAI
import os
import sys
from pathlib import Path
import random
import json
import logging

from langchain.callbacks.base import BaseCallbackHandler

class FullChainCallback(BaseCallbackHandler):
    def __init__(self):
        self.steps = []  # 按时间顺序放所有 log

    def on_agent_action(self, action, **kwargs):
        self.steps.append(action.log)

    def on_tool_end(self, output, **kwargs):
        self.steps.append(f"Observation: {output}")

    def on_agent_finish(self, finish, **kwargs):
        self.steps.append(f"Final Thought: {finish.log}")

os.environ['KMP_DUPLICATE_LIB_OK']='True'



# ============ TOOL DEFINITIONS ============
def classify_image(image_path):
    return ImageClassifierTool().classify_image(image_path)

image_classifier_tool = Tool(
    name="Image Type Classifier",
    func=classify_image,
    description=(
        "输入图像路径，判断图像类型（B超图像或眼底图图像）。"
        "适合用于图像输入后的第一步处理。"
    )
)

def query_rag_with_citation(question):
    result = rag_chain(question)
    answer = result["result"]
    sources = result.get("source_documents", [])
    citations = "\n".join([
        f"来源{i+1}: {doc.metadata.get('source', '未知来源')}" for i, doc in enumerate(sources)
    ])
    return f"答复: {answer}\n\n{citations}"

rag_qa_tool = Tool(
    name="Ophthalmology Knowledge QA",
    func=query_rag_with_citation,
    description=(
        "输入一个眼科相关问题（中文），查询知识库内容并返回详细回答及其引用来源。"
        "适用于眼科疾病定义、症状解释、诊断流程等。"
    )
)

# Tool registry
tools = [fundus_report_tool,ultrasound_report_tool,image_classifier_tool, rag_qa_tool]



callback_manager = CallbackManager([FileCallbackHandler(log_file)])

# ============ LLM for CoT Agent ============
# model_COT = "qwen3"
# model_COT = "gpt4"
model_COT = "deepseek"

if model_COT == "qwen3":
    # ag_llm = ChatOllama(
    #     model="qwen3:8b",
    #     base_url="http://localhost:6006",
    #     temperature=0.2,
    #     top_p=0.3
    # )
    ag_llm = ChatOpenAI(
        model="qwen-plus",  # 可换为 "qwen3-14b-instruct" / "qwen3-72b-instruct" 等
        api_key= "sk-0ac382815f4e4b81a6629812d8e9910d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
        max_tokens=1500

    )
elif model_COT == "gpt4":
    ag_llm = ChatOpenAI(
        model="gpt-4o",  # 也可以换成 gpt-4-turbo / qwen2.5 等
        api_key="",  # 你的代理 key
        base_url="",  # 代理地址
        temperature=0.2,
        max_tokens=1500
    )
elif model_COT == "deepseek":
    ag_llm = ChatOpenAI(
        model="deepseek-chat",  # 也可以换成 gpt-4-turbo / qwen2.5 等
        api_key="",  # 你的代理 key
        base_url="",  # 代理地址
        temperature=0.2,
        max_tokens=1500
    )

# ============ COT Agent Initialization ============
agent_executor = initialize_agent(
    tools=tools,
    llm=ag_llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    callback_manager=callback_manager,
    return_intermediate_steps=True
)

# ============ Entry Function ============
def run_cot_ophthalmology_agent(user_question: str, image_paths: list[str] = []):
    """
    Agent输入为完整提示，Agent将自行调用Tool（CoT风格）完成多步推理。
    """
    prompt = f"You are an experienced assistant ophthalmologist. Your task is to comprehensively analyze the user's questions and the uploaded images, and provide detailed diagnostic suggestions.\n"
    prompt += f"User question: {user_question}\n"
    prompt += f"""
    If the user has uploaded an image, please follow the steps below to proceed: 
    1. List of images uploaded by the user: {image_paths}
    2. For each image, use the Image Type Classifier tool to determine its type (either "fundus image" or "B-ultrasound image"). It is strictly prohibited to confirm the image type based on the naming path of the image! This method is extremely inaccurate and must be confirmed using the tool.
    3. Input the paths of all images classified as "fundus image" into the fundus_report_tool one by one to obtain the report for each image. For example, directly input "D:\\12678754\\M1\\20230613084251_31000901_55637.jpg".
    4. Input the paths of all the images classified as "Ultrasound Images" one by one into the function "ultrasound_report_tool" to obtain the report for each image. For example, directly input "D:\\12678754\\M1\\20230613084251_31000901_55637.jpg".
       For different types of images, different tools are used for analysis to generate medical reports:
        - 眼底图像：The image path (string) must be directly entered into the fundus_report_tool;
        - B超图像：The image path (string) must be directly entered into the ultrasound_report_tool;
        ⚠️ Meaning: Do not input in list format and do not modify the image path. 
        ⚠️ The path of each image is independent and cannot be modified for the incoming image path.
    5. If you need medical knowledge support, please use the Ophthalmology Knowledge QA tool (rag_qa_tool) to search for relevant information, and mark the source of the reference in your answer."""

    prompt += """
    Please **think step by step**, and carry out the reasoning and operations in the following format:
    Thought: (Please describe your thought process here)
    Action: (Invoke tools, such as FundusReportTool, UltrasoundReportTool or KnowledgeQATool)
    Action Input: (The input of the tool, such as a list of image paths or the query content)
    Observation: (The result returned by the tool) 
    ...（It may require multiple rounds of Action/Observation）...  
    Final Answer: (The diagnostic suggestions derived from all the information, detailed in English, including the reasons)

    Please provide the final diagnosis recommendation in English.
    """

    cb = FullChainCallback()
    # result = agent_executor.run(prompt)
    result = agent_executor.invoke({"input": prompt}, callbacks=[cb])

    final_answer = result["output"]  # 最终诊疗建议

    full_think = "\n".join(cb.steps)
    intermediate = result["intermediate_steps"]  # List[(AgentAction, str)]

    # 如果想把 think 串成字符串
    think_chain = "\n".join(
        f"Thought: {action.log}\n"
        f"Action: {action.tool}({action.tool_input})\n"
        f"Observation: {obs}\n"
        for action, obs in intermediate
    )

    return think_chain+full_think+ final_answer

# Example usage
def example():

    json_file = ""


    with open(json_file,'r',encoding='utf-8') as f:
        data = json.load(f)
    for i in range(len(data)):
        image_path = []
        image_path.append(data[i]['fundus'][0])
        image_path.append(data[i]['B-scan'][0])

        print(image_path)


        pro = "Hello, I would like to inquire about some eye-related issues. Please note that I cannot replace a doctor's diagnosis; this is merely for informational purposes. I recently underwent a B-ultrasound image/retinal image examination. Based on the results of my examination, could you please provide me with professional advice?"
        res = ""
        print("=" * 50)
        result = run_cot_ophthalmology_agent(
            user_question=pro,
            image_paths=image_path
        )

        print("\nThe final medical treatment recommendation is:\n", result)

        with open(Path(json_file).parent / (Path(json_file).stem +"_"+ model_COT + f".txt"),'a',encoding='utf-8') as f:
            for s in image_path:
                res = res + s + "\n"
            res = res + result
            f.write(res+"\n\n\n")


if __name__ == "__main__":
    example()
