import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from tqdm import tqdm

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

df = pd.read_excel('/Users/nam/Desktop/iai/my_env/updated_sample_texts.xlsx')

# 요약을 저장할 새로운 컬럼 생성
df['요약 내용'] = ''

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    input_text = row['내용'].replace('\n', ' ')
    raw_input_ids = tokenizer.encode(input_text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
    summary_ids = model.generate(torch.tensor([input_ids]), num_beams=6, min_length=0, max_length=124, eos_token_id=1)
    summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    df.at[idx, '요약 내용'] = summary

# 변경된 데이터프레임을 새로운 엑셀 파일로 저장
df.to_excel('updated_124_sample_texts.xlsx', index=False)
