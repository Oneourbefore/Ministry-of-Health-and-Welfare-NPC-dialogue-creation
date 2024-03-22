import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

def summarize_text(input_text):
    input_text = input_text.replace('\n', ' ')
    raw_input_ids = tokenizer.encode(input_text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
    summary_ids = model.generate(torch.tensor([input_ids]), num_beams=6, min_length=0, max_length=512, eos_token_id=1)
    summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return summary

def greet_and_select_category():
    name = input("1. 이름을 입력하세요: ")
    print(f"반갑습니다 {name}님, 카테고리를 골라주세요.")
    categories = {
        "1": "음악",
        "2": "영화",
        "3": "스포츠",
        "4": "음식",
        "5": "여행",
        "6": "책",
        "7": "게임",
        "8": "패션",
        "9": "기술",
        "10": "예술"
    }
    
    for number, category_name in categories.items():
        print(f"{number}. {category_name}")
    choice = input("원하는 카테고리 번호를 선택하세요: ")
    category_name = categories.get(choice, "알 수 없는 카테고리")
    return choice, category_name

def discuss_category(choice, category_name):
    if category_name != "알 수 없는 카테고리":
        print(f"{category_name}를 골랐습니다. {category_name}에 관해서 50자 이상으로 이야기 해주세요.")
        user_input = input()
        summary = summarize_text(user_input)
        print(f"요약: {summary}\n라는 말씀이시군요~")
    else:
        print("잘못된 카테고리 번호입니다. 다시 시도해주세요.")

choice, category_name = greet_and_select_category()
discuss_category(choice, category_name)
