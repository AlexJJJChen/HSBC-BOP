import pandas as pd

# 读取Excel文件
df_finetune_baichuan = pd.read_excel('baichuan_finetuning_ansewer.xlsx')

# 初始化分数
score_total = 0
score_narrative = 0
score_tf = 0

# 特殊词汇列表
special_words = ["贸易便利试点", "高水平便利试点", "区域便利试点", "区域便利化试点", "特殊离岸", "特殊离岸转手", "境内仓单专卖", "非报关人", "特殊退汇", "前期费用", "退款", "支付机构外汇支付划转"]

for i in range(len(df_finetune_baichuan)):
    # 数据清洗
    query = ''.join(df_finetune_baichuan['query'][i].split('\n'))
    if "follow form "==df_finetune_baichuan['query'][i].split('符合下面的格式：')[-1].split('。')[0]:
        narrative='：'+df_finetune_baichuan['query'][i].split('。帮我判断一下')[0].split("query: 这个是我的交易附言:")[-1].split(':')[-1]
        #print(narrative)
        df_finetune_baichuan['response'][i]=df_finetune_baichuan['response'][i].split('：')[0]+narrative
        #print(df_finetune_baichuan['response'][i])
        
    for word in special_words:
        if word in query:
            #print(i,word,query)
            a=df_finetune_baichuan['response'][i].split(word)
            a.append(word)
            response = ''.join(a).strip()
            #print(response)
            df_finetune_baichuan['response'][i] = response
            #print(i,word,query,response)
    
    # 评估
    if df_finetune_baichuan['ground_truth'][i] == df_finetune_baichuan['response'][i]:
        score_total += 1
    
    ground_truth_narrative = df_finetune_baichuan['ground_truth'][i].split("最终交易附言为：")[-1].split("。")[0]
    response_narrative = df_finetune_baichuan['response'][i].split("最终交易附言为：")[-1].split("。")[0]
    if ground_truth_narrative == response_narrative:
        score_narrative += 1
    else:
        print(i,f"query: {df_finetune_baichuan['query'][i]}\nfinetune answer: {df_finetune_baichuan['response'][i]}\nground truth answer: {df_finetune_baichuan['ground_truth'][i]}\n")

    ground_truth_tf = df_finetune_baichuan['ground_truth'][i].split("最终交易附言为：")[0]
    response_tf = df_finetune_baichuan['response'][i].split("最终交易附言为：")[0]
    if ground_truth_tf == response_tf:
        score_tf += 1

# 打印结果
print("score total:", score_total)
print("score narrative:", (score_narrative) / len(df_finetune_baichuan))
print("score truth/false:", score_tf / len(df_finetune_baichuan))