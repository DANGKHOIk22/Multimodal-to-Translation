import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm

def run_training_loop(model, trainer, tokenizer, ds, get_dataloaders):
    # === Hyperparameters ===
    VOCAB_SIZE = len(tokenizer.get_vocab())
    EMBEDDING_DIMS = 128
    HIDDEN_DIMS = 64
    N_LAYERS = 1
    N_HEADS = 16
    DROPOUT = 0.2
    LR = 0.1
    EPOCHS = 40
    WEIGHT_DECAY = 1e-3
    soft_target_loss_weight = 0.3
    ce_loss_weight = 0.7
    T = 2

    PAD_TOKEN = tokenizer.convert_tokens_to_ids("<pad>")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === Model ===
    model = model(
        VOCAB_SIZE,
        EMBEDDING_DIMS,
        N_HEADS,
        HIDDEN_DIMS,
        N_LAYERS,
        DROPOUT
    ).to(device)

    # === Loss, optimizer, scheduler ===
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    scaler = GradScaler()

    # === Dataloaders ===
    train_dataloader, val_dataloader, _ = get_dataloaders(ds, tokenizer)

    for epoch in range(EPOCHS):
        model.train()
        trainer.model.eval()
        train_losses = []
        train_label_losses = []
        total_train_samples = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_bar = tqdm(train_dataloader, desc="Training", leave=False)

        for idx, samples in enumerate(train_bar):
            try:
                src_seqs = samples['src_seq'].to(device, non_blocking=True)
                tgt_seqs = samples['tgt_seq'].to(device, non_blocking=True)
                src_key_padding_mask = samples['src_key_padding_mask'].to(device, non_blocking=True)
                tgt_key_padding_mask = samples['tgt_key_padding_mask'].to(device, non_blocking=True)

                student_logits = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = trainer.model(
                        input_ids=src_seqs,
                        attention_mask=src_key_padding_mask,
                        decoder_input_ids=tgt_seqs,
                        output_hidden_states=False,
                        output_attentions=False,
                        return_dict=True,
                    ).logits

                label_input = student_logits.permute(0, 2, 1)[:, :, :-1]
                target_labels = tgt_seqs[:, 1:]

                label_loss = criterion(label_input, target_labels)

                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size(0) * (T ** 2)

                loss = ce_loss_weight * label_loss + soft_target_loss_weight * soft_targets_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                batch_size = src_seqs.size(0)
                train_losses.append(loss.item() * batch_size)
                train_label_losses.append(label_loss.item() * batch_size)
                total_train_samples += batch_size

                train_bar.set_postfix(loss=loss.item(), label_loss=label_loss.item())

            except RuntimeError as e:
                print(f"Error in batch {idx}: {e}")
                torch.cuda.empty_cache()
                continue
            finally:
                torch.cuda.empty_cache()

        # === Validation ===
        model.eval()
        val_losses = []
        total_val_samples = 0
        val_bar = tqdm(val_dataloader, desc="Validation", leave=False)

        with torch.no_grad():
            for idx, samples in enumerate(val_bar):
                try:
                    src_seqs = samples['src_seq'].to(device, non_blocking=True)
                    tgt_seqs = samples['tgt_seq'].to(device, non_blocking=True)
                    src_key_padding_mask = samples['src_key_padding_mask'].to(device, non_blocking=True)
                    tgt_key_padding_mask = samples['tgt_key_padding_mask'].to(device, non_blocking=True)

                    student_logits = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)

                    label_input = student_logits.permute(0, 2, 1)[:, :, :-1]
                    target_labels = tgt_seqs[:, 1:]
                    label_loss = criterion(label_input, target_labels)

                    batch_size = src_seqs.size(0)
                    val_losses.append(label_loss.item() * batch_size)
                    total_val_samples += batch_size

                    val_bar.set_postfix(val_loss=label_loss.item())

                except RuntimeError as e:
                    print(f"Validation Error in batch {idx}: {e}")
                    torch.cuda.empty_cache()
                    continue
                finally:
                    torch.cuda.empty_cache()

        avg_train_loss = sum(train_losses) / total_train_samples
        avg_label_loss = sum(train_label_losses) / total_train_samples
        avg_val_loss = sum(val_losses) / total_val_samples

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Label Loss: {avg_label_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step()
