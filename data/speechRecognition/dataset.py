from data.speechRecognition.audio_processing import extract_transcription_and_audio, process_and_augment_audio_features, pad_and_truncate
from data.speechRecognition.text_processing import encode_and_pad_text,preprocess_text
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

# ========== Example Pipeline ==========
def prepare_dataset(minds_dataset, split='train'):
    # Step 1: Extract
    df = extract_transcription_and_audio(minds_dataset, split=split)

    # Step 2: Preprocess text
    df['transcription'] = df['transcription'].apply(preprocess_text)
    df['transcription'] = df['transcription'].apply(lambda x: list(x))

    # Step 3: Augment and extract features
    df_augmented = process_and_augment_audio_features(df,split)


    # Step 4: Determine max lengths
    
    max_text_len = df['transcription'].apply(len).max() + 2

    max_signal_len = df_augmented['audio'].apply(lambda x: x.shape[0]).max()
    #print(df_augmented)
    return df_augmented, max_signal_len, max_text_len

# ========== Dataset Class ==========
class SpeechRecognitionDataset(Dataset):
    def __init__(self, data, char_to_idx, max_signal_len, max_text_len):
        self.data = data
        self.char_to_idx = char_to_idx
        self.max_signal_len = max_signal_len
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data.iloc[idx]['audio']
        text = self.data.iloc[idx]['transcription']
        signal = pad_and_truncate(signal, self.max_signal_len)
        encoder_text,decoder_text, length, attention_mask = encode_and_pad_text(text, self.char_to_idx, self.max_text_len)
        length = torch.tensor(length, dtype=torch.long)
        signal = torch.tensor(signal, dtype=torch.float32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
        encoder_text = torch.tensor(encoder_text, dtype=torch.long)
        decoder_text = torch.tensor(decoder_text, dtype=torch.long)
        return signal, encoder_text,decoder_text , length, attention_mask
    


def get_dataloaders(train_data,val_data,test_data, char_to_idx, max_signal_len, max_text_len):

    train_batch_size = 16
    test_batch_size = 16

    # Creating datasets
    train_dataset = SpeechRecognitionDataset(train_data, char_to_idx, max_signal_len, max_text_len)
    val_dataset = SpeechRecognitionDataset(val_data, char_to_idx, max_signal_len, max_text_len)
    test_dataset = SpeechRecognitionDataset(test_data, char_to_idx, max_signal_len, max_text_len)

    # Creating data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, test_dataloader,test_dataset