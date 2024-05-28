import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# from swift.llm import (
#     get_model_tokenizer, get_template, inference, ModelType,
#     get_default_template_type, inference_stream
# )
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
import torch
import json
import pandas as pd
from swift.tuners import Swift
from tqdm import tqdm

# model_type = ModelType.chatglm3_6b
# template_type = get_default_template_type(model_type)
# print(f'template_type: {template_type}')  # template_type: qwen

ckpt_dir = "output/baichuan2-13b-chat/v6-20240514-101354/checkpoint-1562"
model_type = ModelType.baichuan2_13b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)

kwargs = {}
# kwargs['use_flash_attn'] = True  # 使用flash_attn

# model, tokenizer = get_model_tokenizer(model_type, torch.float16,
#                                        model_kwargs={'device_map': 'auto'}, **kwargs)
# 修改max_new_tokens
# model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)

data_response = []
data_query=[]
data_gt=[]

# 打开 JSON 文件
with open("bopdata_test.json", "r", encoding="utf-8") as json_file:
    # 使用 json.load() 方法加载 JSON 数据
    data = json.load(json_file)

# data 现在包含了 JSON 文件中的内容，通常是一个 Python 字典或列表

print(data)
for i in tqdm(range(len(data))):
    query = f'''{data[i]['query']}
     
    "这个是我的交易附言:"后面的一个句子，注意是一个句子，不是两个句子，代表用户输入的交易附言。
    
    需要注意下面的特殊情况：
    特殊情况一：如果用户输入的交易附言里面包含 "贸易便利试点", "高水平便利试点", "区域便利试点", "区域便利化试点", "特殊离岸", "特殊离岸转手", "境内仓单专卖", "非报关人", "特殊退汇", "前期费用", "退款", "支付机构外汇支付划转"的话，则将这些特殊词汇放到最后。
    这是一个正确的例子："这个是我的交易附言:贸易便利试点一般贸易进口xx。帮我判断一下我的交易附言是否符合下面的格式：一般贸易+货物名称 。并帮我输出正确的交易附言。答案:错误，最终交易附言为：一般贸易xx贸易便利试点"
    这是一个错误的例子："这个是我的交易附言:高水平便利试点一般贸易进口xx。帮我判断一下我的交易附言是否符合下面的格式：一般贸易+货物名称 。并帮我输出正确的交易附言。答案:最终交易附言为：高水平便利试点一般贸易xx" 正确答案应该是 "一般贸易xx高水平便利试点"
    

    特殊情况二：如果用户输入的交易附言里包含"进口"这个词，将这个词去掉。
    这是一个正确的例子："这个是我的交易附言:一般贸易进口xx。帮我判断一下我的交易附言是否符合下面的格式：一般贸易+货物名称  。并帮我输出正确的交易附言。答案:错误，最终交易附言为：一般贸易xx"
    这是一个错误的例子："这个是我的交易附言:一般贸易进口xx。帮我判断一下我的交易附言是否符合下面的格式：一般贸易+货物名称  。并帮我输出正确的交易附言。答案:正确，最终交易附言为：一般贸易进口xx"

    你要看清楚用户输入的交易附言，判断是否符合特殊情况，一定要看清楚，有些时候交易附言里面明明没有"贸易便利试点"的，但你会看错，做一定要再三仔细检查。
    
    '''
    response, history = inference(model, template, query)

    data_query.append(query)
    data_response.append(response)
    data_gt.append(data[i]['response'])
df=pd.DataFrame()
df['query']=data_query
df['response']=data_response
df['ground_truth']=data_gt
df.to_excel('baichuan_finetuning_ansewer.xlsx')


