from sentence_transformers import SentenceTransformer
import torch
import kss
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
# This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

input_text = '''
요즘 직장 생활이 정말 힘들어요. 아침에 일찍 출근해서 밤늦게까지 일하는데도 끝이 없는 것 같아요. 할 일은 계속 쌓이고, 기한은 다가오고, 압박감은 점점 더 커지고 있어요. 사실, 제 자신을 잃어가는 것 같은 느낌도 들어요. 열정적으로 시작했던 일이 이제는 그저 해야 하는 일로만 느껴지네요.
진짜 힘든 건, 이 모든 걸 누구에게도 털어놓지 못한다는 거예요. 가족이나 친구들에게조차 부담을 주고 싶지 않아서, 그냥 혼자 모든 걸 감당하려고 해요. 그런데 그게 점점 더 버거워지는 것 같아요.
가끔은 이 모든 스트레스가 저를 어디로 데려갈지, 정말로 괜찮을 수 있을지 걱정돼요. 하지만 여기 이 자조모임에 와서 이렇게 말할 수 있어서 다행이라고 생각해요. 여러분과 공유할 수 있어서, 조금은 마음이 가벼워지는 것 같아요.
'''

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
input_text = input_text.replace('\n', ' ')
raw_input_ids = tokenizer.encode(input_text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=6, min_length=124, max_length=1024,  eos_token_id=1)
summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

sbert_model = SentenceTransformer('bongsoo/kpf-sbert-v1.1')

def cal_similarity(summary, content):
    summary_embedding = sbert_model.encode(summary, convert_to_tensor=True)
    sen_list = kss.split_sentences(content)
    top_sentences = []
    top_similarities = []

    for sen in sen_list:
        sen_embedding = sbert_model.encode(sen, convert_to_tensor=True)
        cosine_similarity = torch.nn.functional.cosine_similarity(summary_embedding, sen_embedding, dim=0)
        if len(top_sentences) < 3:
            top_sentences.append(sen)
            top_similarities.append(cosine_similarity.item())
        else:
            min_index = top_similarities.index(min(top_similarities))
            if cosine_similarity > top_similarities[min_index]:
                top_sentences[min_index] = sen
                top_similarities[min_index] = cosine_similarity.item()

    return top_sentences

top_similar_sentences = cal_similarity(summary_text, input_text)
print(top_similar_sentences)
