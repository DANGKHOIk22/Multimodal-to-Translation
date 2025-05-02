import re  # Add this import for regular expressions

# ========== Special Tokens ==========
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# ========== Text Preprocessing ==========
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_and_pad_text(text, char_to_idx, max_text_len):
    encoder_labels = [char_to_idx.get(c, 0) for c in text]
    length = len(encoder_labels)

    # Add BOS and EOS tokens
    text_with_tokens = [BOS_TOKEN] + list(text) + [EOS_TOKEN]
    decoder_labels = [char_to_idx.get(c, 0) for c in text_with_tokens]
    attention_mask = [1] * len(decoder_labels)

    # Truncate or pad
    if length > max_text_len:
        encoder_labels = encoder_labels[:max_text_len]
        decoder_labels = decoder_labels[:max_text_len]
        attention_mask = attention_mask[:max_text_len]
    else:
        pad_len_enc = max_text_len - len(encoder_labels)
        pad_len_dec = max_text_len - len(decoder_labels)
        encoder_labels += [0] * pad_len_enc
        decoder_labels += [0] * pad_len_dec
        attention_mask += [0] * pad_len_dec

    return encoder_labels, decoder_labels, length, attention_mask

# ========== Vocabulary ==========
def build_vocab():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    blank_char = "-"
    special_tokens = [BOS_TOKEN, EOS_TOKEN]
    full_vocab = list(vocab + blank_char) + special_tokens

    char_to_idx = {char: idx + 1 for idx, char in enumerate(full_vocab)}  # start from 1
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return char_to_idx, idx_to_char, len(full_vocab)
