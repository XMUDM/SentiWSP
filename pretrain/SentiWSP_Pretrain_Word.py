import os, sys, random
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.nn.functional as F
import datasets
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining, get_linear_schedule_with_warmup
from _utils.utils import *
from _utils.would_like_to_pr import *
from tqdm import tqdm
from transformers import WEIGHTS_NAME, CONFIG_NAME
import pickle
import argparse

parser = argparse.ArgumentParser(description='Pre training model configuration')
parser.add_argument('--model', default='electra',help='pre train model type')
parser.add_argument('--size', default='large',help='pre train model size,choose in [small, base, large]')
parser.add_argument('--dataset', default='wiki',help='choose in [wiki,owt,merg]')
parser.add_argument('--gpu_num', type=int ,default=4,help='pre train gpu num')
parser.add_argument('--pretrain_model', type=str ,default='./pretrain_model/',help='pre train model path')
parser.add_argument('--save_model', type=bool ,default=True,help='Whether to Save model')
parser.add_argument('--save_pretrain_model', type=str ,default='./save_pretrain_model/',help='save will create pre train model path')
parser.add_argument("--rank", type=int,default=-1, help="rank")
parser.add_argument("--local_rank", type=int,default=-1, help="local rank")
parser.add_argument("--sentimask_prob", type=float,default=0.5, help="The sentiment word mask probability")
parser.add_argument('--maskprob', type=float,default = 0.15,help='the mask prob in pretrain process')
parser.add_argument('--batch_size', type=int,default = 64,help='the batch_size in pretrain process')
parser.add_argument('--max_len', type=int,default = 128,help='the seq max_len in pretrain process')
args = parser.parse_args()

# 总体配置
config = MyConfig({
    'base_run_name': 'ELECTRA',
    'seed': 2022,
    'electra_mask_style': True,
    'config_path':'./config_pretrain/',
    'num_workers': 3
})

i = ['small', 'base', 'large'].index(args.size)
config.lr = [2e-5, 1e-5, 1e-5][i]
print("now learning rate:",config.lr)
print("now mask prob",args.mask_prob)
config.max_epoch = 10
config.single_gpunum = 0
#generator_size_divisor = [4, 3, 4][i]

# seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# Global Multi GPU environment definition
if  args.gpu_num > 1:
    #os.environ["CUDA_VISIBLE_DEVICES"] = config.dpp_gp u_num
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda:" + str(config.single_gpunum) if torch.cuda.is_available() else "cpu")
print("want use gpu num:",args.gpu_num)
print("use gpu num:",torch.cuda.device_count())

# load data
if args.dataset=='wiki' or args.dataset=='merg':
    print('load/download wiki dataset')
    if os.path.exists("./datasets/wiki") == False:
        wiki = datasets.load_dataset('wikipedia', '20200501.en', cache_dir='./datasets')['train']
        wiki.save_to_disk("./datasets/wiki")
    else:
        wiki = datasets.load_from_disk("./datasets/wiki")
    print('load/create data from wiki dataset for ELECTRA')
if args.dataset=='owt' or args.dataset=='merg':
    print('load/download OpenWebText Corpus')
    if os.path.exists("./datasets/owt") == False:
        owt = datasets.load_dataset('openwebtext', cache_dir='./datasets')['train']
        owt.save_to_disk('./datasets/owt')
    else:
        owt = datasets.load_from_disk('./datasets/owt')
    print('load/create data from OpenWebText Corpus for ELECTRA')

pretrain_path = args.pretrain_model+args.model+"/"+args.size

if os.path.exists(pretrain_path) == False:
    os.makedirs(pretrain_path)
    print("load Electra Tokenizer Fast from hub...")
    hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{args.size}-discriminator")
    hf_tokenizer.save_pretrained(pretrain_path)
else:
    print("load Electra Tokenizer Fast from pretrain_path...")
    hf_tokenizer = ElectraTokenizerFast.from_pretrained(pretrain_path)
    
ELECTRAProcessor = partial(ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=args.max_length)
dsets = []
if args.dataset=='wiki' or args.dataset=='merg':
    e_wiki = ELECTRAProcessor(wiki).map(cache_file_name=f"./datasets/wiki/electra_wiki_{args.max_length}.arrow", num_proc=8)
    print("wiki len", len(e_wiki))
    dsets.append(e_wiki)
if args.dataset=='owt' or args.dataset=='merg':
    e_owt = ELECTRAProcessor(owt, apply_cleaning=False).map(cache_file_name=f"./datasets/owt/electra_owt_{args.max_length}.arrow", num_proc=8)
    print("owt len",len(e_owt))
    dsets.append(e_owt)

if len(dsets)>1:
    merged_dsets = datasets.concatenate_datasets(dsets)
    print("use merge dataset len:",len(merged_dsets))
elif args.dataset=='wiki':
    merged_dsets=dsets[0]
    print("use wiki dataset len:",len(merged_dsets))
else:
    merged_dsets=dsets[0]
    print("use owt dataset len:",len(merged_dsets))

def get_pad_mask_and_token_type(example):

    # PAD sentence
    if len(example['input_ids']) < args.max_length:
        example['new_input_ids'] = example['input_ids'] + [hf_tokenizer.pad_token_id] * (args.max_length - len(example['input_ids']))
    else:
        example['new_input_ids'] = example['input_ids']
        
    attention_mask = torch.tensor(example['new_input_ids']) != hf_tokenizer.pad_token_id

    # sentence A (token_type_ids =0) sentence B (token_type_ids =1)
    token_type_ids = torch.tensor([0]*example['sentA_length'] + [1]*(args.max_length-example['sentA_length']))
    example['token_type_ids'] = token_type_ids
    example['attention_mask'] = attention_mask
    return

print("pad mask map in"+args.dataset)
if args.dataset=="wiki":
    merged_dsets = merged_dsets.map(get_pad_mask_and_token_type,
                    cache_file_name = f"./datasets/wiki/padmask/electra_wiki_{args.max_length}_padmask_map.arrow",num_proc=16)
elif args.dataset=="owt":
    merged_dsets = merged_dsets.map(get_pad_mask_and_token_type,
                                    cache_file_name = f"./datasets/owt/padmask/electra_owt_{args.max_length}_padmask_map.arrow",num_proc=16)
else:
    merged_dsets = merged_dsets.map(get_pad_mask_and_token_type,
                                    cache_file_name = f"./datasets/merged_dsets/padmask/electra_merg_{args.max_length}_padmask_map.arrow",num_proc=16)

sentivetor = np.load('./sentiment_vocab/senti_vector.npy')

def get_senti_type(example):
    senti_list = []
    for ids in example['new_input_ids']:
        if sentivetor[ids] == 1:
            senti_list.append(1)
        else:
            senti_list.append(0)
    example['senti_type'] = senti_list
    return example


if args.dataset=="wiki":
    merged_dsets = merged_dsets.map(get_senti_type,
                    cache_file_name = f"./datasets/wiki/sentimask/electra_wiki_{args.max_length}_sentimask_map.arrow",num_proc=16)
elif args.dataset=="owt":
    merged_dsets = merged_dsets.map(get_senti_type,
                                    cache_file_name = f"./datasets/owt/sentimask/electra_owt_{args.max_length}_sentimask_map.arrow",num_proc=16)
else:
    merged_dsets = merged_dsets.map(get_senti_type,
                                    cache_file_name = f"./datasets/merged_dsets/sentimask/electra_merg_{args.max_length}_sentimask_map.arrow",num_proc=16)

# mask config
mlm_probability=args.mask_prob
ignore_index=-100
mask_token_index = hf_tokenizer.mask_token_id
special_tok_ids = hf_tokenizer.all_special_ids
vocab_size=hf_tokenizer.vocab_size
replace_prob=0.0 if config.electra_mask_style else 0.1
orginal_prob=0.15 if config.electra_mask_style else 0.1
sentimask_prob = args.sentimask_prob

def mask_tokens_map(example):
    #introduce sentiment mask process
    inputs = torch.tensor(example['new_input_ids']).clone()
    device = inputs.device

    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)

    senti_probability_matrix = torch.tensor(example['senti_type']).clone() * sentimask_prob

    special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)

    for sp_id in special_tok_ids:
        special_tokens_mask = special_tokens_mask | (inputs == sp_id)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    senti_probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    mlm_mask = torch.bernoulli(probability_matrix).bool()
    senti_mask = torch.bernoulli(senti_probability_matrix).bool()

    #  merge mlm mask and senti mask
    mlm_mask = mlm_mask | senti_mask

    labels[~mlm_mask] = ignore_index

    mask_prob = 1 - replace_prob - orginal_prob
    mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
    inputs[mask_token_mask] = mask_token_index

    if int(replace_prob) != 0:
        rep_prob = replace_prob / (replace_prob + orginal_prob)
        replace_token_mask = torch.bernoulli(
            torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

    pass

    example['masked_inputs'] = inputs
    example['is_mlm_applied'] = mlm_mask
    example['labels'] = labels
    return example

print("mlm mask map in"+args.dataset)
if args.dataset=="wiki":
    if args.sentimask_prob == 0:
        #no sentiment mask
        merged_dsets = merged_dsets.map(mask_tokens_map,
                                    cache_file_name = f"./datasets/wiki/mlmmask/electra_wiki_{args.max_length}_padmask_map.arrow",num_proc=16)
    else:
        merged_dsets = merged_dsets.map(mask_tokens_map,
                                    cache_file_name=f"./datasets/wiki/mlmmask/electra_{args.sentimask_prob*10}_wiki_{args.max_length}_mlmmask_map.arrow",
                                    num_proc=16)
elif args.dataset=="owt":
    if args.sentimask_prob == 0:
        merged_dsets = merged_dsets.map(mask_tokens_map,
                                    cache_file_name = f"./datasets/owt/mlmmask/electra_owt_{args.max_length}_padmask_map.arrow",num_proc=16)
    else:
        merged_dsets = merged_dsets.map(mask_tokens_map,
                                    cache_file_name=f"./datasets/owt/mlmmask/electra_{args.sentimask_prob*10}_owt_{args.max_length}_mlmmask_map.arrow",
                                    num_proc=16)
else:
    merged_dsets = merged_dsets.map(mask_tokens_map,
                                    cache_file_name = f"./datasets/merged_dsets/mlmmask/electra_merg_{args.max_length}_mlmmask_map.arrow",num_proc=16)

merged_dsets.set_format(type='torch', columns=['masked_inputs', 'token_type_ids', 'attention_mask','is_mlm_applied','labels'])
print("set format")

# model config
class ELECTRAModel(nn.Module):
  
    # Model initialization needs to be specified generator and discriminator
    def __init__(self, generator, discriminator, hf_tokenizer):
        super().__init__()
        self.generator, self.discriminator = generator,discriminator
        self.hf_tokenizer = hf_tokenizer

    def forward(self, masked_inputs, is_mlm_applied, labels, attention_mask, token_type_ids):
        """
        masked_inputs (Tensor[int]): (batch_size, max_seq_len)
        sentA_lenths (Tensor[int]): (batch_size)
        is_mlm_applied (Tensor[boolean]): (batch_size, max_seq_len), 值为True代表改位置被MASK
        labels (Tensor[int]): (batch_size, max_seq_len), -100 for positions where are not mlm applied
        """
        
        # gen_logits形状 (batch_size, max_seq_len, vocab_size)
        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0]
        # reduce size to save space and speed
        # mlm_gen_logits ：（Mask_num, vocab_size）
        mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)

        with torch.no_grad():
            # sampling
            pred_toks = torch.multinomial(F.softmax(mlm_gen_logits, dim=-1), 1).squeeze()
            
            # Fill the predicted token into the input originally masked as the input of the discriminator
            generated = masked_inputs.clone() # (B,L)
            generated[is_mlm_applied] = pred_toks # (B,L)
            
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone() # (B,L)
            is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def save_discriminator(self, output_dir):
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        model_to_save = self.discriminator
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        print("save model in :" + output_dir)

    def save_generator(self, output_dir):
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        model_to_save = self.generator
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        print("save model in :" + output_dir)

if config.my_model:
    # 训练配置
    config_path=config.config_path

    if os.path.exists(config_path+args.size+'disc') == False:
        print("load ele config in hub...")
        disc_config = ElectraConfig.from_pretrained(f'google/electra-{args.size}-discriminator')
        gen_config = ElectraConfig.from_pretrained(f'google/electra-{args.size}-generator')
        os.makedirs(config_path + args.size+'disc')
        os.makedirs(config_path + args.size+'gen')
        disc_config.save_pretrained(config_path + args.size+'disc')
        gen_config.save_pretrained(config_path + args.size+'gen')
    else:
        print("load ele config in "+config_path)
        disc_config = ElectraConfig.from_pretrained(config_path + args.size+'disc')
        gen_config = ElectraConfig.from_pretrained(config_path + args.size+'gen')

    # note that public electra-small model is actually small++ and don't scale down generator size
    gen_config.hidden_size = int(disc_config.hidden_size/generator_size_divisor)
    gen_config.num_attention_heads = disc_config.num_attention_heads//generator_size_divisor
    gen_config.intermediate_size = disc_config.intermediate_size//generator_size_divisor

    generator = ElectraForMaskedLM(gen_config)
    discriminator = ElectraForPreTraining(disc_config)
else:
    print("use bentchmark generator and discriminator")
    gen_path=args.pretrain_model+args.model+'/'+args.size+'/gen'
    disc_path=args.pretrain_model+args.model+'/'+args.size+'/disc'
    if os.path.exists(gen_path) == False:
        os.makedirs(gen_path)
        os.makedirs(disc_path)
        generator = ElectraForMaskedLM.from_pretrained(f'google/electra-{args.size}-generator')
        discriminator = ElectraForPreTraining.from_pretrained(f'google/electra-{args.size}-discriminator')
        generator.save_pretrained(gen_path)
        discriminator.save_pretrained(disc_path)
    else:
        print("load Generator and disc")
        generator = ElectraForMaskedLM.from_pretrained(gen_path)
        discriminator = ElectraForPreTraining.from_pretrained(disc_path)

discriminator.electra.embeddings = generator.electra.embeddings
generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

electra_model = ELECTRAModel(generator, discriminator, hf_tokenizer)

def ELECTRALoss(pred, targ_ids, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
    gen_loss_fc = nn.CrossEntropyLoss()
    disc_loss_fc = nn.BCEWithLogitsLoss()
    
    mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
    gen_loss = gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
    disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
    is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
    if disc_label_smooth:
      is_replaced = is_replaced.float().masked_fill(~is_replaced, disc_label_smooth)
    disc_loss = disc_loss_fc(disc_logits.float(), is_replaced.float())


    return gen_loss * loss_weights[0] + disc_loss * loss_weights[1],gen_loss,disc_loss

def train(model, dataset, max_epoch):

    if  config.gpu_num > 1:
        sampler = DistributedSampler(dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(params=electra_model.parameters(), lr=config.lr, eps=1e-8, betas=(0.9,0.999), weight_decay=0.01)
    
    total_steps = (dataset.num_rows // args.batch_size) * config.max_epoch
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 1500, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    model.train()
    step=0
    gen_loss_list,disc_loss_list=[],[]
    for epoch in range(max_epoch):
        
        # Each epoch scrambles the data set
        # dataset = dataset.shuffle()
        if args.gpu_num>1:
            sampler.set_epoch(epoch)
        total_train_loss = 0
        for iter_num, batch in enumerate(tqdm(train_loader)):
            pass
            step+=1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.forward(**batch)
            loss,gen_loss,disc_loss = ELECTRALoss(outputs,batch['labels'])
            total_train_loss += loss.item()
            
            loss.backward()
            # clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 200 == 0:
                if dist.get_rank() == 0:
                    gen_loss_list.append(gen_loss.item())
                    disc_loss_list.append(disc_loss.item())
                print(">>> epoth: %d, step: %d, loss: %.4f" % (epoch, step, loss.item()))

            if step ==200:
                if dist.get_rank() == 0:
                    if args.save_model:
                        output_dir = args.save_pretrain_model
                        save_model(model, output_dir, step)
            if step%1000==0 and step>0:
                if dist.get_rank() == 0:
                    if args.save_model:
                        output_dir = args.save_pretrain_model
                        save_model(model, output_dir, step)
                if step==5000:
                    if dist.get_rank() == 0:
                        with open('./loss_word.pkl', 'wb') as f:
                            pickle.dump(gen_loss_list, f, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(disc_loss_list, f, pickle.HIGHEST_PROTOCOL)
                    sys.exit()

        if dist.get_rank() == 0:
            if args.save_model:
                output_dir=args.save_pretrain_model
                save_model(model,output_dir,epoch+1)
        print("Epoch: %d, Average training loss: %.4f" %(epoch, total_train_loss/len(train_loader)))
        
def save_model(model,output_dir,flag):
    if flag>100:
        save_dis_path = output_dir + str(flag) + 'iter_discriminator'
        save_gen_path = output_dir + str(flag) + 'iter_generator'
    else:
        save_dis_path = output_dir + str(flag) + 'epoch_discriminator'
        save_gen_path = output_dir + str(flag) + 'epoch_generator'
    if os.path.exists(save_dis_path) == False:
        os.makedirs(save_dis_path)
        os.makedirs(save_gen_path)
    if args.gpu_num > 1:
        model.module.save_discriminator(save_dis_path)
        model.module.save_generator(save_gen_path)
    else:
        model.save_discriminator(save_dis_path)
        model.save_generator(save_gen_path)

electra_model.to(device)
if  args.gpu_num > 1:
    if torch.cuda.device_count() > 1:
        print("Let's use", args.gpu_num, "GPUs!")
        electra_model = torch.nn.parallel.DistributedDataParallel(electra_model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    else:
        print("There are not enough GPUs available!")
        sys.exit()    
train(electra_model,merged_dsets,config.max_epoch)
