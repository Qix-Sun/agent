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
model_COT = "gpt4"
# model_COT = "deepseek"

if model_COT == "qwen3":
    ag_llm = ChatOpenAI(
        model="qwen-plus",  # 可换为 "qwen3-14b-instruct" / "qwen3-72b-instruct" 等
        api_key= "",
        base_url="",
        temperature=0.2,
        max_tokens=1500
        # extra_body={"enable_thinking": True}  # 是否启用思考模式，可去掉此行
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
    prompt = f"您是一位经验丰富的眼科助理医生。您的任务是全面分析用户的问题以及上传的图像，并给出详细的诊断建议。\n"
    prompt += f"用户问题: {user_question}\n"
    prompt += f"""
    如果用户已上传了图片，请按照以下步骤操作以继续：
    1. 用户上传的图片列表： {image_paths}
    2. 对于每张图像，使用“图像类型分类器”工具来确定其类型（要么是“眼底图像”，要么是“B 超图像”）。严禁根据图像的命名路径来确定其类型！这种方法极其不准确，必须通过该工具来进行确认。
    3. 将所有被分类为“眼底图像”的图片的路径逐一输入到“fundus_report_tool”中，以获取每张图片的报告。例如，直接输入“D：\12678754\M1\20230613084251_31000901_55637.jpg”".
    4. 将所有被分类为“超声图像”的图像的路径逐一输入到“ultrasound_report_tool”函数中，以获取每个图像的报告。例如，直接输入“D：\ 12678754\ M1\ 20230613084251_31000901_55637.jpg”.
       对于不同类型的图像，会使用不同的工具进行分析，从而生成医疗报告：
        - 眼底图像：图像路径（字符串）必须直接输入到眼底报告工具中；
        - B超图像：图像路径（字符串）必须直接输入到“超声报告工具”中；
        ⚠️ 请勿以列表形式输入，也请勿修改图像路径。
        ⚠️ 每张图像的路径都是独立的，对于传入的图像路径无法进行修改。
    5.  如果您需要医学知识方面的帮助，请使用眼科知识问答工具（rag_qa_tool）来查找相关信息，并在您的回答中注明参考来源。"""

    prompt += """
    Please **think step by step**, and carry out the reasoning and operations in the following format:
    Thought: (Please describe your thought process here)
    Action: (Invoke tools, such as FundusReportTool, UltrasoundReportTool or KnowledgeQATool)
    Action Input: (The input of the tool, such as a list of image paths or the query content)
    Observation: (The result returned by the tool) 
    ...（It may require multiple rounds of Action/Observation）...  
    Final Answer: (The diagnostic suggestions derived from all the information, detailed in Chinese, including the reasons)

    Please provide the final diagnosis recommendation in Chinese.
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
    # 开始记录
    # logger = Logger('log/output.log')
    # sys.stdout = logger
    # sys.stderr = logger  # 同时捕获错误输出
    json_file = ""


    with open(json_file,'r',encoding='utf-8') as f:
        data = json.load(f)
    for i in range(len(data)):
        image_path = []
        image_path.append(data[i]['fundus'][0])
        image_path.append(data[i]['B-scan'][0])

        print(image_path)


        pro = "您好，我想咨询一些与眼睛相关的问题。请注意，我不能替代医生的诊断；这只是提供信息的用途。我最近进行了 B 超图像/视网膜图像检查。根据我的检查结果，能否请您给我提供专业的建议？"
        res = ""
        print("=" * 50)
        result = run_cot_ophthalmology_agent(
            user_question=pro,
            image_paths=image_path
        )

        print("\nThe final medical treatment recommendation is:\n", result)

        with open(Path(json_file).parent / (Path(json_file).stem +"_"+ model_COT + f"_cn.txt"),'a',encoding='utf-8') as f:
            for s in image_path:
                res = res + s + "\n"
            res = res + result
            f.write(res+"\n\n\n")

if __name__ == "__main__":
    example()
