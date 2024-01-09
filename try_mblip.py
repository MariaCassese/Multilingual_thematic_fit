# pip install accelerate bitsandbytes
import torch
import requests
from PIL import Image
import transformers
#from transformers import BertTokenizer
#Blip2Processor, Blip2ForConditionalGeneration

"""processor = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Gregor/mblip-mt0-xl", load_in_8bit=True, device_map="auto")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "Describe the image in German."
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.bfloat16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))"""
