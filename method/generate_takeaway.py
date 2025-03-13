import json
import numpy as np
import os
import torch
from torch.utils.data import ConcatDataset
import transformers
from transformers.utils import PaddingStrategy
from openai import OpenAI
import re
from transformers.utils.versions import require_version

from zhipuai import ZhipuAI

#######################################################
#               多agent实现异步对话                    # 
#######################################################
#同步对话在一个python文件里边没办法实现多agengt的交互。必须用异步编程。

glmmode='chatglm3-6b'

gmode='THUDM/chatglm3-6b'

gptmode='gpt-4o'
clientglm = OpenAI(
        
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001)),
    )







#tidx=[1,2,3,4,7]
tidx=[6]
alpha=['b']
for i in range(len(tidx)):
    for j in range(len(alpha)):
        qlist=tidx[i]
        de=alpha[j]
        #file_path = f"../deep/question_cat{qlist}{de}.json"
        #file_path = f"../deep/question_cat{qlist}{de}.json"
        file_path=f"../deep/question_cat{qlist}.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            jsondata=json.load(file)

        tmpdiag=jsondata[0]['dialogue']
        tmpans=jsondata[0]['answer']
        finalrst=[]
        useproblem=jsondata[0]['question']

     
        tmp1="For the following elementary school math questions, please ignore the specific scenarios and background information (such as finance, shopping, life scenarios, etc.), focus only on the essence of mathematics, analyze the characteristics of the question type, and review the problem-solving process. \nSummarize the precautions for solving such questions, as well as the experience that needs to be paid attention to in preventing mistakes."

#Round1:
        messages2=[]
        content2=f" Here is the target problem:{useproblem}\n[dialogue]:{tmpdiag} [answer]:{tmpans}\n{tmp1}"#让glm先知道问题，更利于针对性反思。
        
        messages2.append({"role":"user","content":content2})
        response2=clientglm.chat.completions.create(
            model=glmmode,
            messages=messages2
        )
        tmp2=response2.choices[0].message.content
        #finalrst.append({"student":tmp2})

        print("tmp2:")
        print(tmp2)
        print('----------------------')


#Round2:evaluator llm        
        mesuffix=""""Review the student's summary based on the following three principles and revise it accordingly:
1.Generality: Ensure the summary focuses on the general problem-solving strategy for this type of math problem, rather than the specifics of this question.
2.Mathematical Abstraction: Remove any references to the problem's background knowledge. The summary should strictly emphasize mathematical structures, reasoning steps, and key computational relationships (e.g., identifying objects being calculated, stepwise computation, numerical relationships).
3.Universal Strategy: Extract and articulate the core principles that are transferable to similar problems. The final summary should provide a general method that can be applied broadly to this class of problems.
If the student's summary fails to meet these criteria, rewrite it to align with them while keeping it clear, concise, and mathematically focused."""
     
        content_s=f"[question]:{useproblem}\n[dialogue]:{tmpdiag}\n[Student summary]:{tmp2}\n[Notice]:{mesuffix}."
        med=[{"role":"user","content":content_s}] 
        
        # response_med=clientgpt.chat.completions.create(
        #     model=gptmode,
        #     messages=med
        # )
        


        #################################################################
        # response_med=clientglm.chat.completions.create(
        #     model="glm-4-plus",
        #     messages=med
        # )
        #注意这里的response——med，即第2部分一定要用本地的chatglm
        #################################################################
        response_med=clientglm.chat.completions.create(
            model=glmmode,
            messages=med
        )

        # response_med=clientllama.chat.completions.create(
        #     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        #     messages=med
        # ) 

        # response_med=clientdeepsk.chat.completions.create(
        #     model="deepseek-reasoner",
        #     messages=med,
        #     stream=False
        # )

        # response_med=clientqwen.chat.completions.create(
        #     model="qwen-plus",
        #     messages=med
        # )

        supertmp=response_med.choices[0].message.content
        print("super-medium")
        print(supertmp)
        print('-------------------------')
        finalrst.append({"student":supertmp})
#Round3:
        # messages3=[]
        messages3=[{"role": "system", "content": "You are a helpful assistant."}]

        finalpre="Your task is to summarize the general principles and common methods for solving problems applicable to this type of topic at the end of the conversation based on the students' answers, questions, and the characteristics of the questions themselves. Please make sure that your summary can help students think and solve problems more effectively when they encounter similar problems in the future. Your answer should focus on the core principles rather than the specific questions themselves. After giving the principles, you can appropriately explain how to apply this principle in conjunction with the current problem."

#Round3 Summarizor LLM
        #content3=f"{finalpre}\n[question]:{useproblem}\n[answer]:{tmpans}\n[Socratic dialogue]:{tmpdiag} \n[student summary]:{finalrst}  Please note that the summary should not exceed 3 key points and should not exceed 1000 words."
        content3=f"{finalpre}\n[question]:{useproblem}\n[answer]:{tmpans}\n[Socratic dialogue]:{tmpdiag} \n[student summary]:{finalrst}. Please note that the summary should not exceed 3 key points and should not exceed 1000 words."
        messages3.append({"role":"user","content":content3})

        response3_gpt=clientgpt.chat.completions.create(
            model=gptmode,
            messages=messages3
        )

        # response3_llama=clientllama.chat.completions.create(
        #     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        #     messages=messages3
        # )

        # response3_qwen=clientqwen.chat.completions.create(
        #     model="qwen-plus",
        #     messages=messages3
        # )

        # response3_deepsk=clientdeepsk.chat.completions.create(
        #     model="deepseek-reasoner",
        #     messages=messages3,
        #     stream=False
        # )

        # response3_deepsk=clientdeepsk.chat.completions.create(
        #     model="deepseek-reasoner",
        #     messages=messages3,
        #     stream=False
        # )

        # response3_zhipu=clientzhipu.chat.completions.create(
        #     model="glm-4-plus",
        #     messages=messages3
        # )
        #这里需要修改
        #*************************
        response3=response3_gpt
        #*************************

        tmp3=response3.choices[0].message.content
        
        fine=[]
        fine.append({"teacher_summary":tmp3})
        
        #finalrst.append({"teacher_summary":tmp3})

        print("tmp3:")
        print(tmp3)
        print('------------------------')

        #注意，当需要运行第5，6个问题时，用注释的这个路径。
        #outerpath = f"../takeaway/{qlist}{de}gpt_create.json"
        outerpath=f"../takeaway/{qlist}{de}gpt_create.json"
        

        with open(outerpath, 'w', encoding='utf-8') as file:
            json.dump(fine, file, ensure_ascii=False, indent=4)
