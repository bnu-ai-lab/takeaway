import os
import json
import re
import concurrent.futures
from openai import OpenAI


idx=[1]
alpha=['a']
mode='gpt'
for tidx in idx:
    for de in alpha:
        #file_path=f"../ablation/answer{tidx}{de}{mode}_keypoint2.json"
        #file_path=f"../ablation/answer{tidx}{mode}_ablation.json"
        
        #file_path=f"../ablation/abl/answer{tidx}{de}{mode}_keypoint13.json"
        #file_path=f"../ablation/answer{tidx}{de}{mode}_create.json"
        file_path=f"./answer{tidx}{de}_socnew.json"

        #file_path=f"../socprinciple/answer{tidx}{de}_principle.json"
       # file_path=f"/home/ubuntu/msy/Socratic-takeaway/codes/new/complete/class/soctake/answer{tidx}{de}gpt_soctake.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            jsondata = json.load(file)

        tru=0
        for item in jsondata:
            #这里需要是单纯soc的答案格式
            #if item['soc_ans']==item['answer'] :
            if  item['soc_takeaway_ans']==item['answer'] or item['new-answer']==item['answer'] :
                tru+=1
        acc=tru/len(jsondata)
        print(f"the {tidx}{de} number is:{len(jsondata)}")
        print(f"socratic{tidx}{de}{mode}_create acc is: {acc:.2%}")
        print('-----------------------------------------------------')
