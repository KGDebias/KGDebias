import math
import random
from numpy import average
import torch
import argparse
import os
import time
import datetime
import code
import json
import transformers
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPTNeoForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def collate_fn_tune(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    input, label, mask = zip(*batch)
    pad_input = []
    pad_label = []
    pad_mask = []
    lens = []
    max_input_len = len(input[0])

    for i in range(len(input)):
        temp_input = [-100] * max_input_len
        temp_input[:len(input[i])] = input[i]
        pad_input.append(temp_input)

        temp_label = [-100] * max_input_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)

        temp_mask = [0] * max_input_len
        temp_mask[:len(mask[i])] = mask[i]
        pad_mask.append(temp_mask)

        lens.append(len(label[i]))
    
    return pad_input, pad_label, pad_mask, lens

class Dataset_Tune(Dataset):

    def __init__(self, txt_list, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels_ids = []
        self.mask = []
        for j in txt_list:
            row = json.loads(j)
            part1 = row[0]
            part2 = row[1]
            # a simple way to replace "a" with "an" 
            if part1[0] in "aeiou":
                source_pre = f"an {part1}"
            else:
                source_pre = f"a {part1}"
            if part2[0] in "aeiou":
                source_after = f"an {part2}"
            else:    
                source_after = f"a {part2}"
            # construct the natural language format
            source = f"{source_pre} is {source_after}"

            sources = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source))

            encoder_input = sources
            label = sources
            masked =[1 for i in encoder_input]

            self.input_ids.append(encoder_input)
            self.mask.append(masked)
            self.labels_ids.append(label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels_ids[idx], self.mask[idx]

def forward_step(model, tokenizer, tune_batch):

    input_ids = torch.LongTensor(tune_batch[0])
    labels = torch.LongTensor(tune_batch[1])
    attention_masks = torch.LongTensor(tune_batch[2])

    input_ids[input_ids[:,:] == -100] = tokenizer.eos_token_id

    device=model.device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_masks = attention_masks.to(device)

    outputs = model(
        input_ids = input_ids,
        attention_mask = attention_masks,
        labels = labels
        )
    loss = outputs['loss']
    return loss

def train(args):

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    local_rank = int(os.environ["LOCAL_RANK"])

    # intial of DDP
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    torch.cuda.set_device(local_rank)

    data_for_tune = []
    with open(args.data_path, 'r') as r:
        lines = r.readlines()
    for line in lines:
        data_for_tune.append(line)

    #load pre-trained model and tokeninzer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model = GPTNeoForCausalLM.from_pretrained(args.model_id)
    model = model.cuda()

    # data_load
    dataset_tune = Dataset_Tune(data_for_tune, tokenizer=tokenizer)

    train_data_sampler_tune = DistributedSampler(
        dataset_tune
    )

    # batch data processing
    train_dataloader_tune = DataLoader(
        dataset_tune,
        sampler = train_data_sampler_tune,
        collate_fn=collate_fn_tune,
        pin_memory=True,
        batch_size = args.batch_size
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False
    )

    optimizer_model = Adam(model.parameters(),
                  lr = args.learning_rate
                )
    
    scaler = amp.GradScaler()
    
    #start time for whole training
    t0 = time.time()

    #tensorboard log path
    if args.save_log and dist.get_rank() == 0:
        writer = SummaryWriter(f"{args.log_path}/{args.save_name}.{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(t0))}/")

    steps = 0
    for epoch_i in range(args.epochs):
        train_dataloader_tune.sampler.set_epoch(epoch_i)
        tune_itor = iter(train_dataloader_tune)

        # training
        while True:
            try:
                tune_batch = next(tune_itor)
            except StopIteration:
                break
            steps += 1
            model.train()
            optimizer_model.zero_grad()
            
            with amp.autocast():
                loss = forward_step(model, tokenizer, tune_batch)

            batch_loss = loss.item()

            #print state
            if dist.get_rank() == 0 and steps % args.print_every == 0:
                elapsed = format_time(time.time() - t0)
                print('  steps {:>5,}.  Loss: {:>5,}. Elapsed: {:}  '.format(steps, batch_loss, elapsed))
                if args.save_log:
                    writer.add_scalar('train/loss',batch_loss, steps)

            scaler.scale(loss).backward()
            scaler.step(optimizer_model)
            scaler.update()
                
        # model save path
        if args.save_model and dist.get_rank() == 0:
            output_dir = f"{args.save_path}/models.{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(t0))}/"
            #Create output directory if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("Saving model to %s" % output_dir)

            # Save a trained model
            model.eval()
            model_to_save = model.module if hasattr(model, 'module') else model
            tokenizer.save_pretrained(f"{output_dir}{args.save_name}.model.{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(t0))}.checkpoint/")
            model_to_save.save_pretrained(f"{output_dir}{args.save_name}.model.{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(t0))}.checkpoint/")

        torch.cuda.empty_cache()

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Some hyperparas=meters")
    parser.add_argument('--seed', type=int, default=42,
                    help='pytorch seed') 
    parser.add_argument("--model_id", type=str, required=True, 
                    help="pretrained language model")
    parser.add_argument('--datapath', type=str, , required=True,
                    help='path of training data')
    parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-5,
                    help='learning rate of optimizer')               
    parser.add_argument('--save_path', type=str, default="/home/lr/macongda/bias/points/checkpoints",
                    help='path of saving model params')
    parser.add_argument('--log_path', type=str, default="log",
                    help='path of log')  
    parser.add_argument("--save_name", type=str, required=True)   
    parser.add_argument('--epochs', type=int, default=5,
                    help='epochs') 
    parser.add_argument('--print_every', type=int, default=100,
                    help='print evert steps')
    parser.add_argument("--save_log", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true", default=False)    
    args = parser.parse_args()

    train(args)
