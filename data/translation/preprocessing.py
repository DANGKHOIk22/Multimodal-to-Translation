import re
from model.translation.fine_tunedMbart import train_model

trainer, tokenizer = train_model()

def text_preprocessing(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    text =  text.strip()

    return text

# Apply preprocessing to your dataset
def preprocess_example(example):
    example['en'] = text_preprocessing(example['en'])
    example['vi'] = text_preprocessing(example['vi'])
    return example

def preprocess_function(examples):
    tokenizer.src_lang = "en_XX"
    inputs = tokenizer(
        examples["en"],
        max_length=200,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,  
        return_tensors=None
    )
    tokenizer.tgt_lang = "vi_VN"
    targets = tokenizer(
        examples["vi"],
        max_length=200,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,  
        return_tensors=None
    )