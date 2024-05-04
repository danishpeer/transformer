from pathlib import Path
import torch
from torch import nn
from utils import get_datasets, get_model
from dataset import causal_mask
from config import get_weights_file_path, get_config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import os

def greedy_decode(model, src, src_msk, src_tokenizer, out_tokenizer, seq_len, device):
    sos_index = src_tokenizer.token_to_id('[SOS]')
    eos_index = src_tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(src, src_msk)

   # start with just sos token
    decoder_input = torch.empty((1,1)).fill_(sos_index).type_as(src).to(device)  # [1,1]

    while True:
        if decoder_input.size(1)==seq_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).to(device)  # [1, 1 ,1 ]
        decoder_output = model.decode(decoder_input, encoder_output, src_msk, decoder_mask) # [1,1,d_model]

        # get next token
        out = model.project(decoder_output[:, -1]) # [1,vocab_size]
        _, next_token = torch.max(out, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1,1).fill_(next_token.item()).type_as(src).to(device)
            ],
            dim=1
        )

        if next_token == eos_index:
            break

    return decoder_input.squeeze(0)






def run_validations(model, val_dl, src_tokenizer, out_tokenizer, seq_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()

    count = 0


    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80


    for batch in val_dl:
        count+=1
        encoder_input = batch['encoder_input'].to(device) # [bs, seq_len]
        encoder_mask = batch['encoder_mask'].to(device) # [bs, 1 , 1, seq_len]

        model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, out_tokenizer, seq_len, device)

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = out_tokenizer.decode(model_out.detach().cpu().numpy())

        # Print the source, target and model output
        print_msg('-'*console_width)
        print_msg(f"{f'SOURCE: ':>12}{source_text}")
        print_msg(f"{f'TARGET: ':>12}{target_text}")
        print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

        if count == num_examples:
            print_msg('-'*console_width)
            break




def train_model(config):

    #defining device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    device = torch.device('mps')
    print(f"using device {device}")

    # create the model directory
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    #get data and model
    train_dl, val_dl, src_tokenizer, out_tokenizer = get_datasets(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), out_tokenizer.get_vocab_size()).to(device)

    # kicking off Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)


    # incase of crash, restarting from the previouly trained epoch
    initial_epoch = 0
    global_step  = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    else:
        print("No Preloading of model specified. Hence, Starting from scratch")


    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    num_epochs = config['num_epochs']
    for epoch in range(initial_epoch, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dl, desc=f'Processing epoch {epoch}/{num_epochs}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # [bs, seq_len]
            decoder_input = batch['decoder_input'].to(device) # [bs, seq_len]
            encoder_mask = batch['encoder_mask'].to(device) # [bs, 1, 1 , seq_len]
            decoder_mask = batch['decoder_mask'].to(device) # [bs, 1, seq_len, seq_len]

            encoder_output = model.encode(encoder_input, encoder_mask)  # [bs, seq_len, d_model]
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # [bs, seq_len, d_model]

            out = model.project(decoder_output) # [bs, seq_len, out_vocab_size]

            labels = batch['labels'].to(device) # [bs, seq_len]

            loss = loss_fn(out.view(-1, out_tokenizer.get_vocab_size()), labels.view(-1))

            #update tqdm
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            #write to tensorboard
            writer.add_scalar('train_loss',loss.item(), global_step)
            writer.flush()

            #backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step +=1

        # save the model after each epoch
        run_validations(model, val_dl, src_tokenizer, out_tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                'epoch' : epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step' : global_step,
            },
            model_filename
        )


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)







    









