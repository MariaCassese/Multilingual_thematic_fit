
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import glob
import os
import ipdb

class Multilingual:

    """
    Class that handles translation from English to Italian, French, Spanish, and Russian 
    using the NLLB model and the Helsinki-NLP model
    """

    def __init__(self, language, lang_code, tokenizer, model, model_name, sentence):
        self.language = language
        self.lang_code = lang_code
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.sentence = sentence
        self.translation = self.translate()
        

    def translate(self):
        """
        The method that performs tokenization and translation 
        using either the Helsinki-NLP model or the NLLB model
        """
        try:
 
            if self.model_name =="Helsinki-NLP": #Se il modello è Helsinki-NLP
                translated_tokens = self.model.generate(**self.tokenizer(self.sentence, return_tensors="pt"))
                for t in translated_tokens:
                    translated_text = self.tokenizer.decode(t, skip_special_tokens=True)

            else: #Se invece il modello è NLLB
                inputs = self.tokenizer(self.sentence, return_tensors="pt") 
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
        
        except KeyError:
            return "Unsupported language or model"
    
        


def main():

    """
    Main method:
    Imports data (datasets EventsAdapt, EventsRev, and DTFit)
    and models (NLLB and Helsinki-NLP)
    and applies the 'translate' method of the class Multilingual
    
    """
    data_dir = r"C:\Users\Utente\Documents\GitHub\Multilingual_thematic_fit\data"
    txt_files = list(glob.glob(f"{data_dir}/*.txt"))
 
    models = {
        "6OOM_model": (
            AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"),
            AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        ),
        "1.3B_model": (
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

    languages = {  #Dictionary that registers the language code IDs for the NLLB model
        "Italian": "ita_Latn",
        "French": "fra_Latn",
        "Spanish": "spa_Latn",
        "Russian": "rus_Cyrl"
    }
    for filename in txt_files:
        print(f"Processing file: {filename}")
        dataset = pd.read_csv(filename, sep="\t", header=None)

        #to use the NLLB model

        for model_name, model_info in models.items():
            print(f"Using {model_name}...")
            if isinstance(model_info, tuple):
                tokenizer, model = model_info
                for language, lang_code in languages.items():
                    print(f"Translating into {language}...")
                    result = {"sentences": []}
                    for idx, row in enumerate(dataset.itertuples()):
                        sentence = row[2]
                        multilingual = Multilingual(
                            language, lang_code, tokenizer, model, model_name, sentence
                        )
                        result["sentences"].append(multilingual.translation)
                    print(f"Translation into {language} completed.")

                    df_result = pd.DataFrame(result)
                    out_file_name = os.path.join(
                        "txt_results",
                        f"{language}_{model_name}_{os.path.basename(filename)}_sentence.txt"
                    )
                    df_result.to_csv(
                        out_file_name,
                        sep="\t",
                        header=None,
                        index=None
                    )

                print(f"Processing of file {filename} using {model_name} completed.\n")
            
            # to use the Helsinki model

            else:  
                for lang_name, lang_info in model_info.items():
                    tokenizer, model = lang_info

                    print(f"Translating into {lang_name}...")
                    result = {"sentences": []}
                    for idx, row in enumerate(dataset.itertuples()):
                        sentence = row[2]
                        multilingual = Multilingual(
                            language=lang_name, 
                            lang_code=None, 
                            tokenizer=tokenizer, 
                            model=model, 
                            model_name=model_name, 
                            sentence=sentence
                        )
                        result["sentences"].append(multilingual.translation)
                        print(f"Translation into {language} completed.")

                    df_result = pd.DataFrame(result)
                    out_file_name = os.path.join(
                        "txt_results",
                        f"{language}_{model_name}_{os.path.basename(filename)}_sentence.txt"
                    )
                    df_result.to_csv(
                        out_file_name,
                        sep="\t",
                        header=None,
                        index=None
                    )
            
                    print(f"Processing of file {filename} using {model_name} completed.\n")

 
if __name__ == "__main__":
    main()

