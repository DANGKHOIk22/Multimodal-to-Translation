import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer, MBartForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)

# ========== Constants ==========
CHECKPOINT = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TGT_LANG = "vi_VN"
MAX_LEN = 200
BATCH_SIZE = 2
NUM_EPOCHS = 2


# ========== Text Preprocessing ==========
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_example(example):
    example['en'] = text_preprocessing(example['en'])
    example['vi'] = text_preprocessing(example['vi'])
    return example


# ========== Tokenization ==========
def preprocess_function(examples, tokenizer):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(examples["en"], max_length=MAX_LEN, truncation=True)

    tokenizer.tgt_lang = TGT_LANG
    labels = tokenizer(examples["vi"], max_length=MAX_LEN, truncation=True)

    inputs["labels"] = labels["input_ids"]
    return inputs


# ========== Trainer Setup ==========
def train_model():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = MBartForConditionalGeneration.from_pretrained(CHECKPOINT)

    # Load and preprocess dataset
    ds = load_dataset("thainq107/iwslt2015-en-vi")
    ds['train'] = ds['train'].map(preprocess_example)
    ds['validation'] = ds['validation'].map(preprocess_example)
    ds['test'] = ds['test'].map(preprocess_example)

    # Tokenize dataset
    tokenized_ds = ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./mbart_EnglishToVietnamese/",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=5e-5,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        report_to="none",
        save_strategy="no",
        save_steps=0,
        save_total_limit=0
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        tokenizer=tokenizer
    )

    # Start training
    trainer.train()

    return trainer, tokenized_ds


# ========== Entry Point ==========
if __name__ == "__main__":
    trainer, tokenized_ds = train_model()
