import os
import json
import re
import concurrent.futures
from openai import OpenAI

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
        print(f"An error occurred: {e}")
        return None

def process_question(data, exmque, exmdiag, exmans, client, tmpmode):
    tmpq = data['question']
    cleaned_answer = str(data['answer']).replace(',', '')   
    orians = int(cleaned_answer)
    #tmpidx = data['index']
    

    #Socratic prompt 模板。
    content = f"Examples: \n[Problem ]:{exmque} \n[answer]:{exmans}\n[dialogue]:{exmdiag}\n\n###[target problem]:{tmpq}  Please analyze and solve the target problem and give the answer."
    
    messages = [{"role": "user", "content": content}]
    result = client.chat.completions.create(messages=messages, model=tmpmode, temperature=0.0, top_p=0.8)
    tmp = result.choices[0].message.content
    
    socans = extract_answer(tmp)
    return {"question": tmpq, "soc_ans": socans if socans is not None else tmp, "answer": orians,"solution":tmp}, socans == orians

def main():
    tmpmode = 'chatglm3-6b'
    client = OpenAI(
     
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001)),
    )
    bldex=[1]
    alpha=['b']
    for tidx in bldex:
        for de in alpha:
            file_path = f"../deep/question_cat{tidx}{de}.json"
            #当tidx为5或者6时
            #file_path=f"../deep/question_cat{tidx}.json"
            with open(file_path, 'r', encoding='utf-8') as file:
                jsondata = json.load(file)

            exmque, exmdiag, exmans = jsondata[0]['question'], jsondata[0]['dialogue'], jsondata[0]['answer']

            soc_orilist = []
            tru = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(process_question, data, exmque, exmdiag, exmans, client, tmpmode) for data in jsondata]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result, correct = future.result()
                    soc_orilist.append(result)
                    if correct:
                        tru += 1
                    print(f"Processed {i + 1}/{len(jsondata)}")

            acc = tru / len(jsondata)
            #print(f"original socratic{tidx}{de} acc is: {acc:.2%}")
            print(f"original socratic{tidx}{de} acc is: {acc:.2%}")
            print('-----------------------------------------------------')

            outerpath = f"./answer{tidx}{de}_socnew.json"
            with open(outerpath, 'w', encoding='utf-8') as file:
                json.dump(soc_orilist, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()


#注意Socraticdiag对话，需要经过evaluation中的extra_answer.py和calculate_acc.py才能得到最终准确结果。

