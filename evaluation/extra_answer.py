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

clientzhipu = ZhipuAI(api_key="")

def extract_answer(text):
    try:
        answer_match = re.search(r"### answer:\s*([\d.]+)", text)
        if answer_match:
            return int(float(answer_match.group(1)))

        the_answer_match = re.search(r"The answer is:\s*([\d.]+)", text, re.IGNORECASE)
        if the_answer_match:
            return int(float(the_answer_match.group(1)))

        last_number_match = re.findall(r"\d+\.?\d*", text)
        if last_number_match:
            return int(float(last_number_match[-1]))

        return None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None


tidx=[1]
alpha=['c']
for i in range(len(tidx)):
    for j in range(len(alpha)):
        qlist=tidx[i]
        de=alpha[j]
        #file_path = f"../ablation/answer{qlist}{de}gpt_create.json"
        #file_path = f"../deep/question_cat{qlist}{de}.json"
        #file_path=f"../soctake/answer{qlist}{de}gpt_soctake.json"
        file_path=f"../soc/answer{qlist}{de}_socnew.json"
        #file_path=f"../soc/answer{qlist}_socnew.json"
        #file_path=f"./answer{qlist}{de}_principle.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            jsondata=json.load(file)
        for a in range(len(jsondata)):
            cor=[]
            tmpsuffix='Please extract the final answer to this question. You only need to give an integer, nothing else is required.'
            tmpq=jsondata[a]['question']
            tmpsolution=jsondata[a]['solution']
            tmpcontent=f"[question]:{tmpq} \n{tmpsolution} {tmpsuffix}"
            cor.append({"role":"user","content":tmpcontent})
            tmpresponse=clientzhipu.chat.completions.create( model="glm-4-flashx",messages=cor)

            tmpans=tmpresponse.choices[0].message.content
            answer=extract_answer(tmpans)

            jsondata[a]['new-answer']=answer
            if answer==jsondata[a]['answer']:
                jsondata[a]['result']=True
            else:
                #动态调整，如果要测takeaway的准确率，就用注释的这一条
                #if jsondata[a]['answer']==jsondata[a]['soc_takeaway_ans']:

                #当前测试的为单纯socratic对话
                if jsondata[a]['answer']==jsondata[a]['soc_ans']:
                    jsondata[a]['new-answer']=jsondata[a]['answer']
                jsondata[a]['result']=False
            print(a)
            print('--------------------------------')
        print('************************************')
        

        #计算总体准确率。经过api和extract函数双重验证
        tru=0
        for i in range(len(jsondata)):
            if  jsondata[a]['new-answer']==jsondata[a]['answer']:
                tru+=1
        acc = tru / len(jsondata)
        formatted_percent = f"{acc:.2%}"
        print(f"question{tidx}{de}_socratic_acc is: {formatted_percent}")
        
        outerpath=file_path
        with open(outerpath, 'w', encoding='utf-8') as file:
            json.dump(jsondata, file, ensure_ascii=False, indent=4)
        
