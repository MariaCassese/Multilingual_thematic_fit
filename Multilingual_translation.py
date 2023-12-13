
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import glob
import os
import ipdb

class Multilingual:

    def __init__(self, language, lang_code, tokenizer, model, sentence):
        self.language = language
        self.lang_code = lang_code
        self.tokenizer = tokenizer
        self.model = model
        self.sentence = sentence
        self.translation = self.translate()
        

    def translate(self):
        inputs = self.tokenizer(self.sentence, return_tensors="pt")
        if 
        translated_tokens = self.model.generate(
            **inputs, 
            forced_bos_token_id = self.tokenizer.lang_code_to_id[self.lang_code],
            max_length=30
        )
        translated_text = self.tokenizer.decode(
            translated_tokens[0], 
            skip_special_tokens=True
        )
        
        return translated_text
    
        


def main():
    data_dir = r"C:\Users\Utente\Documents\GitHub\Multilingual_thematic_fit\data"
    txt_files = list(glob.glob(f"{data_dir}/*.txt"))
 
    models = {
        "6OOM model": (
            AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"),
            AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        ),
        "1.3B model": (
            AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B"),
            AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
        ),
        "Helsinki-NLP": {
            "Italian": (
                AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it"),
                AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
            ),
            "French": (
                AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr"),
                AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            ),
            "Spanish":(
                AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
                AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
            ),
            "Russian":(
                AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru"),
                AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
            ),
        },
    }

    """languages = {
        "Italian": (
            tokenizer.lang_code_to_id["ita_Latn"]
        ),
        "French": (
            tokenizer.lang_code_to_id["fra_Latn"]
        ),
        "Spanish": (
            tokenizer.lang_code_to_id["spa_Latn"]
        ),
        "Russian": (
            tokenizer.lang_code_to_id["rus_Cyrl"]
            
        ),
    }"""

    languages = {
        "Italian": "ita_Latn",
        "French": "fra_Latn",
        "Spanish": "spa_Latn",
        "Russian": "rus_Cyrl"
    }
    for filename in txt_files:
        print(f"Processing file: {filename}")  
        dataset = pd.read_csv(filename, sep="\t", header=None)
        for model_name in models.items():
            for language, (lang_code) in list(languages.items()):
                for model_name, (tokenizer, model) in models.items():
                    print(f"Translating into {language}...")
                    print(f"Using {model_name} for {language}:")
                    result = {
                        "sentences": [],
                    }
                    for idx, row in enumerate(dataset.itertuples()):
                        sentence = row[2]           
                        multilingual = Multilingual(
                            language, lang_code, tokenizer, model, sentence
                        )
                        result["sentences"].append(multilingual.translation)
                        print(f"Translation into {language} completed.")
            
                    df_result = pd.DataFrame(result)
                    out_file_name = os.path.join(
                        "txt_results", f"{language}_{model_name}_{os.path.basename(filename)}_sentence.txt"
                        )
                    df_result.to_csv(
                        out_file_name, 
                        sep="\t", 
                        header=None, 
                        index=None
                    )

            print(f"Processing of file {filename} completed.\n")    
 
if __name__ == "__main__":
    main()
