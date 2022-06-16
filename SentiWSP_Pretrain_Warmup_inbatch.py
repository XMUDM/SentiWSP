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
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining, get_linear_schedule_with_warmup, ElectraModel
from _utils.utils import *
from _utils.would_like_to_pr import *
from tqdm import tqdm
from transformers import WEIGHTS_NAME, CONFIG_NAME
import argparse


parser = argparse.ArgumentParser(description='Pre training model configuration')
parser.add_argument('--model', default='electra',help='pre train model type')
parser.add_argument('--size', default='small',help='pre train model size,choose in [small, base, large]')
parser.add_argument('--dataset', default='wiki',help='choose in [wiki,owt,merg]')
parser.add_argument('--gpu_num', type=int ,default=4,help='pre train gpu num')
parser.add_argument('--load_model', type=str,default='word_level5',help='continue train model path')
parser.add_argument('--pretrain_model', type=str,default='./pretrain_model/',help='pre train model path')
parser.add_argument("--rank", type=int,default=-1, help="rank")
parser.add_argument("--local_rank", type=int,default=-1, help="local rank")
parser.add_argument('--batch_size', type=int,default = 64,help='the batch_size in pretrain process')
parser.add_argument('--max_len', type=int,default = 128,help='the seq max_len in pretrain process')
parser.add_argument('--save_pretrain_model', type=str ,default='./save_pretrain_model/',help='save will create pre train model path')
parser.add_argument('--Negative_type', type=str,default = 'random',help='[random,generate]the negative sample creat way')
parser.add_argument('--random_type', type=str,default = 'sentivocab',help='[sentivocab,allvocab]the random replace in which vocab')
parser.add_argument("--sentimask_prob", type=float,default=0.5, help="The sentiment word mask probability")
parser.add_argument("--train_type", type=str,default='unsup', help="[unsup or sup]]")
parser.add_argument('--save_model', type=bool ,default=True,help='Whether to Save model')
parser.add_argument('--use_jsondata', type=bool ,default=True,help='use my jsondata')
parser.add_argument('--jsondata_path', type=str ,default="./datasets/wikijson/wiki_50w_20%.json",help='json data path')
args = parser.parse_args()


# The default configuration is config. If there are no special requirements, it does not need to be changed
config = MyConfig({
    'base_run_name': 'ELECTRA',
    'seed': 2022,
    'electra_mask_style': True,
    'config_path':'./config_pretrain/',
    'num_workers': 3
})
i = ['small', 'base', 'large'].index(args.size)
config.mask_prob = [0.15, 0.15, 0.25][i]
config.lr = [2e-5, 1e-5, 1e-5][i]
config.max_epoch = 10
config.single_gpunum = 0
generator_size_divisor = [4, 3, 4][i]
#  countinue train model path
config.model_path = args.pretrain_model+args.load_model+'/'+args.size+'/disc'
print("load raw model in :",config.model_path)
config.pooling = 'cls'
config.temperature = 0.05
# If Negative_Type select the generator population, and specify the generator path here
config.generator_path = args.pretrain_model+args.load_model+'/'+args.size+'/gen'


print("train type:",args.train_type)
print("now learning rate:",config.lr)
if args.train_type=='sup':
    print("Negative generation way:",args.Negative_type)
    if args.Negative_type=='random':
        print("random way in:", args.random_type)
else:
    print("negative sample in batch")

# seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# mult GPU
if  args.gpu_num > 1:
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.dpp_gpu_num
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda:" + str(config.single_gpunum) if torch.cuda.is_available() else "cpu")


if args.use_jsondata == False:
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

    ELECTRAProcessor = partial(ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=args.max_len)
    dsets = []
    if args.dataset=='wiki' or args.dataset=='merg':
        e_wiki = ELECTRAProcessor(wiki).map(cache_file_name=f"./datasets/wiki/electra_wiki_{args.max_len}.arrow", num_proc=8)
        print("wiki len", len(e_wiki))
        dsets.append(e_wiki)
    if args.dataset=='owt' or args.dataset=='merg':
        e_owt = ELECTRAProcessor(owt, apply_cleaning=False).map(cache_file_name=f"./datasets/owt/electra_owt_{args.max_len}.arrow", num_proc=8)
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

    '''
    Original sentence retention data e_wiki_org (example 'input_ids')
    '''
    #e_wiki_org = e_wiki
    pass

    '''
    Data of emotional words by percentage mask sentimask_input_ids
    '''
    sentivetor = np.load('./sentiment_vocab/senti_vector.npy')
    mask_token_index = hf_tokenizer.mask_token_id
    special_tok_ids = hf_tokenizer.all_special_ids
    vocab_size=hf_tokenizer.vocab_size

    def get_senti_mask(example):
        senti_list = []

        new_input_ids = torch.tensor(example['input_ids']).clone()

        for ids in example['input_ids']:
            if sentivetor[ids] == 1:
                senti_list.append(1)
            else:
                senti_list.append(0)

        senti_probability_matrix = torch.tensor(senti_list).clone() * args.sentimask_prob
        senti_mask = torch.bernoulli(senti_probability_matrix).bool()
        new_input_ids[senti_mask] = mask_token_index
        example['sentimask_ids'] = new_input_ids

        return example
    print("senti mask map in"+args.dataset)
    if args.dataset=="wiki":
        merged_dsets = merged_dsets.map(get_senti_mask,
                        cache_file_name = f"./datasets/wiki/sentence/sentimask/electra_wiki_{args.sentimask_prob}_{args.max_len}_sentimask_map.arrow",num_proc=16)
    elif args.dataset=="owt":
        merged_dsets = merged_dsets.map(get_senti_mask,
                        cache_file_name = f"./datasets/owt/sentence/sentimask/electra_owt_{args.sentimask_prob}_{args.max_len}_sentimask_map.arrow",num_proc=16)
    else:
        merged_dsets = merged_dsets.map(get_senti_mask,
                        cache_file_name = f"./datasets/merged_dsets/sentence/sentimask/electra_merg_{args.sentimask_prob}_{args.max_len}_sentimask_map.arrow",num_proc=16)



    '''
    e_wiki_org、e_wiki_sentimask  pad、attention_mask、token_type_ids
    '''
    def get_org_sentimask_pad_mask_and_token_type(example):

        # PAD sentence
        if len(example['input_ids']) < args.max_len:
            example['ori_input_ids'] = example['input_ids'] + [hf_tokenizer.pad_token_id] * (args.max_len - len(example['input_ids']))
        else:
            example['ori_input_ids'] = example['input_ids']

        if len(example['sentimask_ids']) < args.max_len:
            example['sentimask_input_ids'] = example['sentimask_ids'] + [hf_tokenizer.pad_token_id] * (args.max_len - len(example['sentimask_ids']))
        else:
            example['sentimask_input_ids'] = example['sentimask_ids']

        attention_mask = torch.tensor(example['ori_input_ids']) != hf_tokenizer.pad_token_id

        # sentence A (token_type_ids =0) sentence B (token_type_ids =1)
        token_type_ids = torch.tensor([0]*example['sentA_length'] + [1]*(args.max_len-example['sentA_length']))
        example['token_type_ids'] = token_type_ids
        example['attention_mask'] = attention_mask
        return example

    print("pad mask map in"+args.dataset)
    if args.dataset=="wiki":
        merged_dsets = merged_dsets.map(get_org_sentimask_pad_mask_and_token_type,
                        cache_file_name = f"./datasets/wiki/sentence/padmask/electra_wiki_{args.sentimask_prob}_{args.max_len}_padmask_map.arrow",num_proc=16)
    elif args.dataset=="owt":
        merged_dsets = merged_dsets.map(get_org_sentimask_pad_mask_and_token_type,
                        cache_file_name = f"./datasets/owt/sentence/padmask/electra_owt_{args.sentimask_prob}_{args.max_len}_padmask_map.arrow",num_proc=16)
    else:
        merged_dsets = merged_dsets.map(get_org_sentimask_pad_mask_and_token_type,
                        cache_file_name = f"./datasets/merged_dsets/sentence/padmask/electra_merg_{args.sentimask_prob}_{args.max_len}_padmask_map.arrow",num_proc=16)

elif args.use_jsondata == True:
    print("use json data file in " + args.jsondata_path)
    merged_dsets = datasets.dataset_dict.DatasetDict.from_json(args.jsondata_path)
else:
    sys.exit()
# Supervised learning generates negative samples
if args.train_type == 'sup':
    senti_vocab = np.load('./sentiment/senti_vocab.npy')
    '''
    Generate data of alternative words randomly or by generator e_wiki_rep
    '''
    if args.Negative_type == 'random':

        def get_replace_word(example):

            rep_input_ids = []
            input_ids = example['sentimask_input_ids'][:]

            for ids in input_ids:
                if ids == mask_token_index:
                    if args.random_type == 'allvocab':
                        rep_input_ids.append(random.randint(0,vocab_size-1))
                    elif args.random_type == 'sentivocab':
                        rep_input_ids.append(senti_vocab[random.randint(0,len(senti_vocab)-1)])
                else:
                    rep_input_ids.append(ids)

            # PAD
            if len(rep_input_ids) < args.max_len:
                example['rep_input_ids'] = rep_input_ids + [hf_tokenizer.pad_token_id] * (args.max_len - len(rep_input_ids))
            else:
                example['rep_input_ids'] = rep_input_ids

            return example
        print("replace in random")
        print("replace map in" + args.dataset)
        if args.dataset == "wiki":
            merged_dsets = merged_dsets.map(get_replace_word,
                            cache_file_name=f"./datasets/wiki/sentence/repword/electra_wiki_{args.sentimask_prob}_{args.max_len}_{args.random_type}_repword_map.arrow",
                                            num_proc=16)
        elif args.dataset == "owt":
            merged_dsets = merged_dsets.map(get_replace_word,
                            cache_file_name=f"./datasets/owt/sentence/repword/electra_owt_{args.sentimask_prob}_{args.max_len}_{args.random_type}_repword_map.arrow",
                                            num_proc=16)
        else:
            merged_dsets = merged_dsets.map(get_replace_word,
                            cache_file_name=f"./datasets/merged_dsets/sentence/repword/electra_merg_{args.sentimask_prob}_{args.max_len}_{args.random_type}_repword_map.arrow",
                                            num_proc=16)
        merged_dsets.set_format(type='torch', columns=['ori_input_ids', 'sentimask_input_ids','rep_input_ids','token_type_ids', 'attention_mask'])

    elif args.Negative_type == 'generate':
        print("replace in model train generator")
        merged_dsets.set_format(type='torch', columns=['ori_input_ids', 'sentimask_input_ids','token_type_ids', 'attention_mask'])

    else:
        print("no negative type  name " + args.Negative_type)
        sys.exit()

elif args.train_type == 'unsup':

    merged_dsets.set_format(type='torch', columns=['ori_input_ids', 'sentimask_input_ids', 'attention_mask'])

else:
    print("no simcse train type name " + args.train_type)
    sys.exit()


class SimcseModel(nn.Module):

    def __init__(self, pretrained_model, model_type='unsup', pooling='cls'):
        super(SimcseModel, self).__init__()
        self.DModel = pretrained_model
        self.pooling = pooling
        self.model_type = model_type

    '''
    ori_input_ids、sentimask_input_ids, rep_input_ids的形状均为：
    (batch_size,max_seq_len)
    '''

    def forward(self, ori_input_ids, pos_input_ids, attention_mask, token_type_ids=None, neg_input_ids=None):

        if self.model_type == 'sup':
            # merge
            input_ids = torch.stack([ori_input_ids, pos_input_ids, neg_input_ids], dim=1)
            input_ids = input_ids.view(args.batch_size * 3, -1).to(device)

            # attention_mask and token_type_ids copy 3 times
            attention_mask = torch.stack([attention_mask, attention_mask, attention_mask], dim=1)
            attention_mask = attention_mask.view(args.batch_size * 3, -1).to(device)
            # token_type_ids = torch.stack([token_type_ids, token_type_ids, token_type_ids],dim=1)
            # token_type_ids = token_type_ids.view(args.batch_size * 3, -1).to(device)

        elif self.model_type == 'unsup':
            # merge
            input_ids = torch.stack([ori_input_ids, pos_input_ids], dim=1)
            input_ids = input_ids.view(args.batch_size * 2, -1).to(device)

            # attention_mask and token_type_ids copy 2 times
            attention_mask = torch.stack([attention_mask, attention_mask], dim=1)
            attention_mask = attention_mask.view(args.batch_size * 2, -1).to(device)
            # token_type_ids = torch.stack([token_type_ids, token_type_ids],dim=1)
            # token_type_ids = token_type_ids.view(args.batch_size * 2, -1).to(device)

        out = self.DModel(input_ids, attention_mask)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [3 * batch_size, hidden_size]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [3 * batch_size, hidden_size, seq_len]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [3 * batch_size, hidden_size]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [3 * batch_size, hidden_size, seq_len]
            last = out.hidden_states[-1].transpose(1, 2)  # [3 * batch_size, hidden_size, seq_len]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [3 * batch_size, hidden_size]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [3 * batch_size, hidden_size]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [3 * batch_size,2, hidden_size]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [3 * batch_size, hidden_size]

    def save_DModel(self, output_dir):
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        model_to_save = self.DModel
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        print("save model in :" + output_dir)


# import model
if args.Negative_type == 'generate' and args.train_type == 'sup':
    rep_generator = ElectraForMaskedLM.from_pretrained(config.generator_path)
    rep_generator.generator_lm_head.weight = rep_generator.electra.embeddings.word_embeddings.weight
    
DModel = ElectraModel.from_pretrained(config.model_path)
SimCSEModel = SimcseModel(pretrained_model=DModel, model_type = args.train_type,pooling=config.pooling)


# simcse Supervised loss function
def simcse_sup_loss(y_pred, temp=0.05):
    """Supervised loss function
    y_pred (tensor): electra output, [batch_size * 3, hidden_size]

    """
    y_true = torch.arange(y_pred.shape[0], device=device)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = torch.index_select(sim, 0, use_row)
    sim = sim / temp
    loss = F.cross_entropy(sim, y_true)
    return loss


# simcse Unsupervised loss function
def simcse_unsup_loss(y_pred, temp=0.05):
    """Unsupervised loss function
    y_pred (tensor): electra output, [batch_size * 2, hidden_size]

    """
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp
    loss = F.cross_entropy(sim, y_true)
    return loss


def train(model, dataset, max_epoch, generator = None):
    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, eps=1e-8, betas=(0.9,0.999), weight_decay=0.01)
    
    total_steps = (dataset.num_rows // args.batch_size) * config.max_epoch
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 1500, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    model.train()
    step=0
    for epoch in range(max_epoch):
        
        if  args.gpu_num > 1:
            sampler = DistributedSampler(dataset)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,sampler=sampler)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        if args.gpu_num > 1:
            sampler.set_epoch(epoch)
        total_train_loss = 0
        
        for iter_num, batch in enumerate(tqdm(train_loader)):
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.train_type == 'sup':
                if args.Negative_type == 'generate':
                    print("data generateing ")
                    inputs = batch['sentimask_input_ids'].clone().to(device)
                    attention_mask = batch['attention_mask'].clone().to(device)
                    token_type_ids = batch['token_type_ids'].clone().to(device)
                    is_sentimask_applied = batch['sentimask_input_ids'] == mask_token_index
                    is_sentimask_applied = is_sentimask_applied.to(device)
                    gen_logits = generator(inputs, attention_mask, token_type_ids)[0]
                    sentimask_gen_logits = gen_logits[is_sentimask_applied, :]
                    pred_toks = torch.multinomial(F.softmax(sentimask_gen_logits, dim=-1), 1).squeeze()
                    rep_input_ids = inputs.clone()
                    rep_input_ids[is_sentimask_applied] = pred_toks
                    rep_input_ids.to(device)
                    outputs = model.forward(ori_input_ids = batch['sentimask_input_ids'],
                                        pos_input_ids = batch['ori_input_ids'],
                                        neg_input_ids = rep_input_ids,
                                        attention_mask = batch['attention_mask'])
                else:
                    outputs = model.forward(ori_input_ids = batch['sentimask_input_ids'],
                                            pos_input_ids = batch['ori_input_ids'],
                                            neg_input_ids = batch['rep_input_ids'],
                                            attention_mask = batch['attention_mask'])
                loss = simcse_sup_loss(outputs, temp=config.temperature)
            elif args.train_type == 'unsup':
                outputs = model.forward(ori_input_ids=batch['sentimask_input_ids'],
                                        pos_input_ids=batch['ori_input_ids'],
                                        attention_mask=batch['attention_mask'])

                loss = simcse_unsup_loss(outputs, temp=config.temperature)

            else:
                print("no simcse type name " + args.train_type)
                sys.exit()
            total_train_loss += loss.item()
            loss.backward()
            # clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                #print(batch['labels'])
                #print(torch.argmax(outputs.logits, 1))
                print(">>> epoth: %d, iter_num: %d, loss: %.4f" % (epoch, step, loss.item()))
            if step == 200:
                if dist.get_rank() == 0:
                    if args.save_model:
                        output_dir = args.save_pretrain_model
                        save_model(model, output_dir, step)
            if step % 1000 == 0 and step > 0:
                if dist.get_rank() == 0:
                    if args.save_model:
                        output_dir = args.save_pretrain_model
                        save_model(model, output_dir, step)
                    if step == 10000:
                        sys.exit()

        if dist.get_rank() == 0:
            if args.save_model:
                output_dir = args.save_pretrain_model
                save_model(model, output_dir, epoch + 1)
            
        print("Epoch: %d, Average training loss: %.4f" %(epoch, total_train_loss/len(train_loader)))
        
def save_model(model,output_dir,flag):
    if flag>100:
        save_dis_path=output_dir+str(flag)+'iter_discriminator'
    else:
        save_dis_path = output_dir + str(flag) + 'epoch_discriminator'
    if os.path.exists(save_dis_path) == False:
        os.makedirs(save_dis_path)
    if args.gpu_num > 1:
        model.module.save_DModel(save_dis_path)
    else:
        model.save_DModel(save_dis_path)
SimCSEModel.to(device)
if args.Negative_type == 'generate'and args.train_type == 'sup':
    rep_generator.to(device)
    
if  args.gpu_num > 1:
    if torch.cuda.device_count() > 1:
        print("Let's use", args.gpu_num, "GPUs!")
        SimCSEModel = torch.nn.parallel.DistributedDataParallel(SimCSEModel,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
        if args.Negative_type == 'generate' and args.train_type == 'sup':
            rep_generator = torch.nn.parallel.DistributedDataParallel(rep_generator,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank)
    else:
        print("There are not enough GPUs available!")
        sys.exit()
        
if args.Negative_type == 'generate'and args.train_type == 'sup':
    train(SimCSEModel, merged_dsets, config.max_epoch, rep_generator)
else:
    train(SimCSEModel, merged_dsets, config.max_epoch)
