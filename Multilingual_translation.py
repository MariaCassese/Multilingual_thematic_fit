
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import glob
import os

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
        translated_tokens = self.model.generate(
            **inputs, 
            forced_bos_token_id = self.lang_code,
            max_length=30
        )
        translated_text = self.tokenizer.decode(
            translated_tokens[0], 
            skip_special_tokens=True
        )
        
        return translated_text
    
        


def main():
    data_dir = r"C:\Users\Utente\Documents\GitHub\Event_Knowledge_Model_Comparison\datasets\id_verbs"
    txt_files = list(glob.glob(f"{data_dir}/*.txt"))
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    languages = {
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
    }
    for filename in txt_files:
        print(f"Processing file: {filename}")  
        dataset = pd.read_csv(filename, sep="\t", header=None)
        for language, (lang_code) in list(languages.items()):
            print(f"Translating into {language}...")
            result = {
                "sentences": [],
                }
            out_file_name = os.path.join(
                    "txt_results", f"{language}_{os.path.basename(filename)}_sentence.txt"
                )
            for idx, row in enumerate(dataset.itertuples()):
                sentence = row[2]           
                multilingual = Multilingual(
                    language, lang_code, tokenizer, model, sentence
                )
                result["sentences"].append(multilingual.translation)
                #print(result)
            # Creazione di un DataFrame per i risultati di questa lingua
            df_result = pd.DataFrame(result)
        
            # Costruzione del nome del file di output
            out_file_name = os.path.join(
                "txt_results", f"{language}_{os.path.basename(filename)}_sentence.txt"
            )

        # Salvataggio del DataFrame nel file di output per questa lingua
        df_result.to_csv(
            out_file_name, 
            sep="\t", 
            header=None, 
            index=None
        )
        print(f"Translation into {language} completed.")

        print(f"Processing of file {filename} completed.\n")    
 
if __name__ == "__main__":
    main()
