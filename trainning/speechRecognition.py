from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from data.speechRecognition.dataset import get_dataloaders

# ========== Evaluation Function ==========
def evaluate(model, dataloader, criterion_encoder, criterion_decoder, device, alpha=0.6):
    model.eval()
    losses = []

    with torch.no_grad():
        dataloader_tqdm = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, encoder_labels, decoder_labels, labels_len, attention_masks in dataloader_tqdm:
            inputs = inputs.to(device)
            encoder_labels = encoder_labels.to(device)
            decoder_labels = decoder_labels.to(device)
            labels_len = labels_len.to(device)
            attention_masks = attention_masks.to(device)

            encoder_outputs, decoder_outputs = model(inputs, decoder_labels, attention_masks)

            logits_lens = torch.full(
                size=(encoder_outputs.size(1),),
                fill_value=encoder_outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss_encoder = criterion_encoder(encoder_outputs, encoder_labels, logits_lens, labels_len)
            loss_decoder = criterion_decoder(decoder_outputs[:, :, :-1], decoder_labels[:, 1:])
            loss = alpha * loss_encoder + (1 - alpha) * loss_decoder

            losses.append(loss.item())
            dataloader_tqdm.set_postfix(loss=loss.item())

    return sum(losses) / len(losses)


# ========== Training Function ==========
def fit(model, train_loader, val_loader, criterion_encoder, criterion_decoder,
        optimizer, scheduler, device, epochs, alpha=0.6, max_grad_norm=2):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []

        model.train()
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for idx, (inputs, encoder_labels, decoder_labels, labels_len, attention_masks) in enumerate(train_loader_tqdm):
            inputs = inputs.to(device)
            encoder_labels = encoder_labels.to(device)
            decoder_labels = decoder_labels.to(device)
            labels_len = labels_len.to(device)
            attention_masks = attention_masks.to(device)

            optimizer.zero_grad()
            encoder_outputs, decoder_outputs = model(inputs, decoder_labels, attention_masks)

            logits_lens = torch.full(
                size=(encoder_outputs.size(1),),
                fill_value=encoder_outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss_encoder = criterion_encoder(encoder_outputs, encoder_labels, logits_lens, labels_len)
            loss_decoder = criterion_decoder(decoder_outputs[:, :, :-1], decoder_labels[:, 1:])
            loss = alpha * loss_encoder + (1 - alpha) * loss_decoder

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_train_losses.append(loss.item())
            train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion_encoder, criterion_decoder, device, alpha)
        val_losses.append(val_loss)

        print(f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    return train_losses, val_losses


# ========== LR Lambda Scheduler ==========
def lr_lambda(epoch, warmup_epochs=2, total_epochs=20, init_scale=0.1, min_scale=0.3):
    scale_range = 1.0 - min_scale

    if epoch < warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        return init_scale + (1.0 - init_scale) * warmup_factor

    decay_factor = (total_epochs - epoch) / (total_epochs - warmup_epochs)
    return min_scale + scale_range * max(0.0, decay_factor)


# ========== Run Training ==========
def run_training(train_data, val_data, test_data, char_to_idx,
                 max_signal_len, max_text_len, vocab_size,
                 DeepSpeech):

    hidden_size = 256
    n_layers = 3
    n_heads = 8
    dropout_prob = 0.2
    unfreeze_layers = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 20
    lr = 1e-3
    weight_decay = 1e-4
    alpha = 0.5
    blank_char = "-"

    model = DeepSpeech(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout_prob,
        unfreeze_layers=unfreeze_layers
    ).to(device)

    criterion_encoder = nn.CTCLoss(
        blank=char_to_idx[blank_char],
        zero_infinity=True,
        reduction="mean",
    )
    criterion_decoder = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_dataloader, val_dataloader, test_dataloader, test_dataset = get_dataloaders(
        train_data, val_data, test_data, char_to_idx, max_signal_len, max_text_len
    )

    train_losses, val_losses = fit(
        model, train_dataloader, val_dataloader,
        criterion_encoder, criterion_decoder,
        optimizer, scheduler, device, epochs, alpha
    )

    return model, train_losses, val_losses, test_dataloader, test_dataset
