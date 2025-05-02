import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ========== Sample Creation ==========
def create_samples(data, tokenizer):
    target_lang_code = "vi_VN"
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]

    all_src_seqs = []
    all_src_key_padding_mask = []
    all_tgt_seqs = []
    all_tgt_key_padding_mask = []

    for sample in data:
        src_encoded = sample['src_ids']
        tgt_encoded = sample['tgt_ids']
        src_padding = sample['src_key_padding_mask']
        tgt_padding = sample['tgt_key_padding_mask']

        src_seq = torch.tensor(src_encoded, dtype=torch.long)
        tgt_seq = torch.tensor(tgt_encoded, dtype=torch.long)

        src_key_padding_mask = torch.tensor(src_padding, dtype=torch.float)
        tgt_key_padding_mask = torch.tensor(tgt_padding, dtype=torch.float)

        all_src_seqs.append(src_seq)
        all_src_key_padding_mask.append(src_key_padding_mask)
        all_tgt_seqs.append(tgt_seq)
        all_tgt_key_padding_mask.append(tgt_key_padding_mask)

    return all_src_seqs, all_src_key_padding_mask, all_tgt_seqs, all_tgt_key_padding_mask


# ========== Dataset Definition ==========
class MTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.src_seqs, self.src_key_padding_mask, self.tgt_seqs, self.tgt_key_padding_mask = create_samples(data, tokenizer)

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return {
            "src_seq": self.src_seqs[idx],
            "src_key_padding_mask": self.src_key_padding_mask[idx],
            "tgt_seq": self.tgt_seqs[idx],
            "tgt_key_padding_mask": self.tgt_key_padding_mask[idx],
        }


# ========== DataLoader Builder ==========
def get_dataloaders(data, tokenizer, train_batch_size=4, test_batch_size=4, num_workers=4):
    train_dataset = MTDataset(data['train'], tokenizer)
    val_dataset = MTDataset(data['validation'], tokenizer)
    test_dataset = MTDataset(data['test'], tokenizer)

    def build_loader(dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    train_loader = build_loader(train_dataset, train_batch_size)
    val_loader = build_loader(val_dataset, test_batch_size)
    test_loader = build_loader(test_dataset, test_batch_size)

    return train_loader, val_loader, test_loader
