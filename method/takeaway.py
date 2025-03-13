import os
import json
import re
import concurrent.futures
from openai import OpenAI

#OpenAI 客户端
tmpmode = 'chatglm3-6b'
client = OpenAI(
    api_key="0",
    base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001)),
)


# 提取答案的函数
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

# 读取 JSON 文件

mode='gpt'

bldex=[1]
alpha=['b']
for tidx in bldex:
    for de in alpha:
        takeawaypath = f"../takeaway/{tidx}{de}{mode}_create.json"
        file_path = f"../deep/question_cat{tidx}{de}.json"

        # takeawaypath = f"../takeaway-test/takeaway_{tidx}{mode}plus.json"
        # file_path = f"../deep/question_cat{tidx}.json"

        with open(takeawaypath, 'r', encoding='utf-8') as file:
            takeaway = json.load(file)

        with open(file_path, 'r', encoding='utf-8') as file:
            jsondata = json.load(file)

        # 示例数据
        exmque = jsondata[0]['question']
        exmdiag = jsondata[0]['dialogue']
        exmans = jsondata[0]['answer']

        # 处理单个问题的函数
        def process_question(data, exmque, exmdiag, exmans, client, tmpmode):
            try:
                tmpq = data['question']
                cleaned_answer = str(data['answer']).replace(',', '')
                orians = int(cleaned_answer)
                
                # content = (
                #     f"Examples: \n[Problem]: {exmque} \n[dialogue]: {exmdiag}\n"
                #     f"[summary]: {takeaway}\n###answer: {exmans}\n\n"
                #     f"###[target problem]: {tmpq} Please analyze and solve the target problem and give the answer."
                # )

                content = (
                    f"Examples: \n[Problem]: {exmque} \n[dialogue]: {exmdiag}\n"
                    f"[summary]: {takeaway}\n###answer: {exmans}\n\n"
                    f"###[target problem]: {tmpq}. Note that you do not need to simulate a teacher-student dialogue during the problem-solving process. Please analyze and solve the target problem and give the answer."
                )
                
                
                messages = [{"role": "user", "content": content}]
                result = client.chat.completions.create(messages=messages, model=tmpmode, temperature=0.0, top_p=0.8)
                tmp = result.choices[0].message.content
                
                socans = extract_answer(tmp)
                if socans==orians:
                    flag=True
                else:
                    flag=False
                return {"question": tmpq, "soc_takeaway_ans": socans if socans is not None else tmp, "answer": orians,"result":flag,"solution":tmp}, socans == orians
            
            except Exception as e:
                print(f"Error processing question: {e}")
                return None, False

        # 处理问题并实时反馈进度
        soc_orilist = []
        tru = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_question, data, exmque, exmdiag, exmans, client, tmpmode): data for data in jsondata}
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result, correct = future.result()
                if result:
                    soc_orilist.append(result)
                    if correct:
                        tru += 1
                
                print(f"Processed {i + 1}/{len(jsondata)}")

        # 计算准确率
        acc = tru / len(jsondata)
        formatted_percent = f"{acc:.2%}"
        if tidx==5 or tidx==6:
            print(f"question{tidx}{mode}_ablation_acc is: {formatted_percent}")
            print('----------------------------------------------------')
            outerpath = f"../ablation/answer{tidx}{mode}_create.json"
        else:
            print(f"question{tidx}{de}{mode}_create_acc is: {formatted_percent}")
            print('----------------------------------------------------')
            outerpath = f"../ablation/answer{tidx}{de}{mode}_create.json"
#注意要经过evaluation文件夹的extra.py和calculate.py