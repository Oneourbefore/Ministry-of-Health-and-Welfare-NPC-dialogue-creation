import torch
from transformers import pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration, TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

class SpellChecker:
    def __init__(self):
        self.corrector = pipeline("text2text-generation", model="j5ng/et5-typos-corrector")

    def correct(self, text):
        corrected_text = self.corrector(text)[0]['generated_text']
        return corrected_text

class KoBARTSummarizer:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

    def summarize(self, input_text):
        input_text = input_text.replace('\n', ' ')
        raw_input_ids = self.tokenizer.encode(input_text)
        input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]
        summary_ids = self.model.generate(torch.tensor([input_ids]), num_beams=6, min_length=0, max_length=512, eos_token_id=1)
        summary = self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        return summary
    
class SentimentAnalyzer:
    def __init__(self):
        model_name = 'hun3359/klue-bert-base-sentiment'  # 감정분석 모델 불러오기, 한국어 데이터셋으로 학습되어 60가지의 감정 라벨을 가짐
        self.analyzer = pipeline('sentiment-analysis', model=model_name)

    def analyze(self, text):
        result = self.analyzer(text)[0]
        sentiment = result['label']
        score = result['score']
        return sentiment, score

class ToxicityChecker:
    def __init__(self):
        model_name = 'smilegate-ai/kor_unsmile'
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=-1,  # cpu: -1, gpu: gpu number
            return_all_scores=True,
            function_to_apply='sigmoid'
        )

    def check(self, text):
        scores = self.pipe(text)[0]
        toxicity_score = next((score['score'] for score in scores if score['label'] == 'toxic'), 0)
        return toxicity_score

class UserInteraction:
    def __init__(self):
        self.categories = {
            "1": "음악",
            "2": "영화",
        }
        self.summarizer = KoBARTSummarizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.spell_checker = SpellChecker()
        self.toxicity_checker = ToxicityChecker()

    def greet_and_select_category(self):
        name = input("1. 이름을 입력하세요: ")
        print(f"반갑습니다 {name}님, 카테고리를 골라주세요.")
        for number, category_name in self.categories.items():
            print(f"{number}. {category_name}")
        choice = input("원하는 카테고리 번호를 선택하세요: ")
        category_name = self.categories.get(choice, "알 수 없는 카테고리")
        return name, choice, category_name

    def discuss_category(self, name, choice, category_name):
        if category_name != "알 수 없는 카테고리":
            while True:
                print(f"{category_name}에 대해 이야기 해주세요. (10자 이상)")
                user_input = input()
                if len(user_input) < 10:
                    print("10자 이상으로 입력해 주세요.")
                    continue
                if self.toxicity_checker.check(user_input) > 0.5:
                    print("위험한 단어가 포함되어 있습니다. 요약하지 않고 멈춥니다.")
                    return
                summary = self.summarizer.summarize(user_input)  # 입력된 텍스트를 요약
                corrected_summary = self.spell_checker.correct(summary)  # 요약된 텍스트를 맞춤법 검사
                # 만약 요약된 결과가 "내가"로 시작한다면 사용자의 이름으로 교체
                if corrected_summary.startswith("내가"):
                    corrected_summary = corrected_summary.replace("내가", f"{name}이(가)")
                sentiment, score = self.sentiment_analyzer.analyze(corrected_summary)  # 수정된 텍스트의 감정 분석
                print(f"{corrected_summary},의 경험을 하셨군요. {sentiment}의 감정이 들었을 것 같아요.")
        else:
            print("잘못된 카테고리 번호입니다. 다시 시도해주세요.")

interaction = UserInteraction()
name, choice, category_name = interaction.greet_and_select_category()
interaction.discuss_category(name, choice, category_name)
