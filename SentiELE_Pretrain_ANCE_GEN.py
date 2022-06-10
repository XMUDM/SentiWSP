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
import json
import faiss
import argparse
from transformers import WEIGHTS_NAME, CONFIG_NAME

# parser config
parser = argparse.ArgumentParser(description='ANN data generate configuration')
parser.add_argument('--model', default='electra',help='pre train model type')
parser.add_argument('--pretrain_model', type=str,default='./pretrain_model/',help='pre train model path')
parser.add_argument('--model_path', default='largesen',help='model path required to generate Ann data（Warm up model）')
parser.add_argument('--tokenizer_path', default='./pretrain_model/electra/large/',help='tokenizer path required to generate Ann data')
parser.add_argument('--size', type=str,default='large',help='the size of model')
parser.add_argument('--max_length', type=int,default = 128,help='the max_length of input sequence')
parser.add_argument('--use_exist_data', type=bool ,default = False,help='Whether to use the sampled data. If not, regenerate it')
parser.add_argument('--data_path', default='./ANCEdata/',help='Using the existing data to generate ANN')
parser.add_argument('--sentimask_prob', type=float ,default = 0.7 ,help='Proportion of emotional words mask')
parser.add_argument('--pooling', default ='cls' ,help='pooling method')
parser.add_argument('--ANN_topK', type=int ,default = 100 ,help='Number of similar documents returned when Ann search')
#parser.add_argument('--negative_num', type=int ,default = 10 ,help='Total number of hard negative selected')
parser.add_argument('--negative_num', type=list ,default = 7 ,help='Total number of hard negative selected')
parser.add_argument('--neg_ann_name', default='sen',help='hard negative path')
parser.add_argument('--gpu_num',default='0',help='Number of GPUs used in training')
parser.add_argument('--batch_size',type=int ,default=64,help='the batch size of emb process')
parser.add_argument('--wiki_num',type=int ,default=500000,help='the sample size of Document')
parser.add_argument('--loadwikijson',type=bool ,default=True,help='the sample size of Document')
args = parser.parse_args()


# train on device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# seed
random_seed = 2022
random.seed(random_seed)
torch.manual_seed(random_seed)

tokenizer_path = args.pretrain_model+args.model+"/"+args.size
model_path=args.pretrain_model+args.model_path
# Tokenizer and settings for loading models
disc_config = ElectraConfig.from_pretrained(model_path)

if os.path.exists(tokenizer_path) == False:
    print("load Electra Tokenizer Fast from hub...")
    hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{args.size}-discriminator")
    hf_tokenizer.save_pretrained(tokenizer_path)
else:
    print("load Electra Tokenizer Fast from tokenizer_path...")
    hf_tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_path)

    
'''
Data of emotional words by percentage mask sentimask_input_ids
'''
sentivetor = np.load('./sentiment_vocab/senti_vector.npy')
mask_token_index = hf_tokenizer.mask_token_id
special_tok_ids = hf_tokenizer.all_special_ids
vocab_size=hf_tokenizer.vocab_size

json_wiki_path=args.data_path+'wikijson/wiki_50w_20%_clean.json'
ann_data=args.data_path+args.model_path+'_'+args.size+str(args.sentimask_prob)+'/wiki'+str(args.sentimask_prob)+'_ANN.json'
ann_data_path=args.data_path+args.model_path+'_'+args.size+str(args.sentimask_prob)
neg_ann_data_path=args.data_path+args.model_path+'_'+args.size+str(args.sentimask_prob)
neg_ann_data=args.data_path+args.model_path+'_'+args.size+str(args.sentimask_prob)+'/wiki_ANN_'+args.neg_ann_name+str(args.negative_num)+'_neg.npy'

def get_senti_mask(example):
    newexample = {}
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
    new_input_ids = new_input_ids.tolist()
    
    newexample['sentence_len'] = len(example['input_ids'])
    
    if len(example['input_ids']) < args.max_length:
        newexample['positive'] = example['input_ids'] + [hf_tokenizer.pad_token_id] * (args.max_length - len(example['input_ids']))
    else:
        newexample['positive'] = example['input_ids']
        
    if len(new_input_ids) < args.max_length:
        newexample['query'] = new_input_ids + [hf_tokenizer.pad_token_id] * (args.max_length - len(new_input_ids))
    else:
        newexample['query'] = new_input_ids
    
    return newexample
    
    
def SampleData():
    if args.loadwikijson:
        e_wiki = datasets.dataset_dict.DatasetDict.from_json(json_wiki_path)
        jsonWiki = []
        print("read wiki json!")
        for index,data in enumerate(e_wiki):
            if index == args.wiki_num:
                break
            new_data={}
            new_data['input_ids']=data['ori_input_ids']
            jsonWiki.append(new_data)
            print("index:",index)
        print("read wiki json down!")
        senti_jsonWiki = []

        for data in jsonWiki:
            newdata = get_senti_mask(data)
            senti_jsonWiki.append(newdata)
    else:
        print('load/download wiki dataset')
        if os.path.exists("./datasets/wiki") == False:
            wiki = datasets.load_dataset('wikipedia', '20200501.en', cache_dir='./datasets')['train']
            wiki.save_to_disk("./datasets/wiki")
        else:
            wiki = datasets.load_from_disk("./datasets/wiki")
        print('load/create data from wiki dataset for ELECTRA')
        ELECTRAProcessor = partial(ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=args.max_length)
        e_wiki = ELECTRAProcessor(wiki).map(cache_file_name=f"electra_wiki_{args.max_length}.arrow", num_proc=8)

    # Sampling method: take the first 500000 items here, and modify them as you like
        jsonWiki = []

        for index,data in enumerate(e_wiki):
            if index == args.wiki_num:
                break
            jsonWiki.append(data)

        senti_jsonWiki = []

        for data in jsonWiki:
            newdata = get_senti_mask(data)
            senti_jsonWiki.append(newdata)
    print("write ann file")
    if os.path.exists(ann_data_path) == False:
        os.makedirs(ann_data_path)
    for data in senti_jsonWiki:
        with open(ann_data, 'a+', encoding='utf-8') as f_obj:
            json_str = json.dumps(data, ensure_ascii=False)
            f_obj.write(json_str + '\n')

        
        
'''
ANN model definition
'''
class ELEDIC_NLL_LN(nn.Module):

    def __init__(self, pretrained_model):
        super(ELEDIC_NLL_LN, self).__init__()
        self.DModel = pretrained_model
        self.norm = nn.LayerNorm(disc_config.hidden_size)

    def get_emb(self, input_ids, attention_mask, pooling='cls'):
        out = self.DModel(input_ids=input_ids,
                                attention_mask=attention_mask)
        # Decide how to get the vector after the query passes through the model
        if pooling == 'cls':
            return self.norm(out.last_hidden_state[:, 0])  # [batch_size, hidden_size]
        
        if pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch_size, hidden_size, seq_len]
            return self.norm(torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1))      # [batch_size, hidden_size]
        
        if pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch_size, hidden_size, seq_len]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch_size, hidden_size, seq_len]                  
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch_size, hidden_size]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch_size, hidden_size]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch_size,2, hidden_size]
            return self.norm(torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1))   # [batch_size, hidden_size]
    
    def forward(self,query_ids,attention_mask_q,
                input_ids_pos,attention_mask_pos,
                input_ids_neg,attention_mask_neg):
        
        '''
        sentence a positive_passage
        sentence b negative_passage
        '''
        q_embs = self.get_emb(query_ids, attention_mask_q)
        pos_embs = self.get_emb(input_ids_pos, attention_mask_pos)
        neg_embs = self.get_emb(input_ids_neg, attention_mask_neg)
        logit_matrix = torch.cat([(q_embs * pos_embs).sum(-1).unsqueeze(1),
                                  (q_embs * neg_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)
    
    def save_DModel(self,output_dir):
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        model_to_save = self.DModel
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        print("save model in :"+output_dir)
        
    
class ANN_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_file,read_data_fn):

        self.example_list = read_data_fn(data_file)
        self.size = len(self.example_list)

    def __getitem__(self, index):
        item = self.example_list[index]
        attention_mask = [1] * item['sentence_len'] + [0] * (args.max_length - item['sentence_len'])
        # Convert list to torch type because dataloader only accepts torch type data
        input_id = torch.tensor(item['input_id'], dtype=torch.int)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        return input_id, attention_mask

    def __len__(self):
        return self.size
    
def InferenceEmbedding(annmodel,train_dataloader,pooling = 'cls'):
    print("***** Running ANN Embedding Inference *****")
    embedding = []
    annmodel.eval()
    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            
            #ignore token_type_ids
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "pooling":pooling
            }
            # Select different EMB methods according to query Q or text P, which should be the same here
            embs = annmodel.get_emb(**inputs)

        embs = embs.detach().cpu().numpy()
        embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    return embedding

def GenerateNegativePassaageID(ANN_Index,negative_sample_num):
    query_negative_passage = {}
    
    for query_idx in range(ANN_Index.shape[0]):

        # The index of the POS document is equal to the index of the query
        pos_pid = query_idx
        top_ann_pid = ANN_Index[query_idx, :].copy()

        query_negative_passage[query_idx] = []

        neg_cnt = 0

        for neg_pid in top_ann_pid:
            # Skip if positive example is detected
            if neg_pid == pos_pid:
                continue
            
            if neg_cnt >= negative_sample_num:
                break

            query_negative_passage[query_idx].append(neg_pid)
            neg_cnt += 1

    return query_negative_passage
    

if __name__ == "__main__":
    if os.path.exists(ann_data) == False:
        SampleData()
    else:
        print("data already in",ann_data)
    # load model
    print("starting generator ann data...")
    print("start generate ann data use checkpoint in "+ model_path)
    discmodel = ElectraModel.from_pretrained(model_path)
    annmodel = ELEDIC_NLL_LN(discmodel)
    annmodel.to(device)
    
    # inference query emb
    print("***** inference of query *****")
    def get_query_data(filename):
        example_list = []
        for line in open(filename, 'rb'):
            example_dict = {}
            row = json.loads(line)
            example_dict['input_id'] = row['query']
            example_dict['sentence_len'] = row['sentence_len']
            example_list.append(example_dict)
        return example_list

    query_dataset = ANN_Dataset(ann_data, get_query_data)
    query_dataloader = torch.utils.data.DataLoader(query_dataset,batch_size=args.batch_size)
    query_embedding = InferenceEmbedding(annmodel, query_dataloader, args.pooling)
    print("***** Done query inference *****")
    
    # inference passages emb
    print("***** inference of passages *****")
    def get_passages_data(filename):
        example_list = []
        for line in open(filename, 'rb'):
            example_dict = {}
            row = json.loads(line)
            example_dict['input_id'] = row['positive']
            example_dict['sentence_len'] = row['sentence_len']
            example_list.append(example_dict)
        return example_list

    passages_dataset = ANN_Dataset(ann_data, get_passages_data)
    passages_dataloader = torch.utils.data.DataLoader(passages_dataset,batch_size=args.batch_size)
    passages_embedding = InferenceEmbedding(annmodel, passages_dataloader, args.pooling)
    print("***** Done passage inference *****")
    
    # Building Ann index to find TOPK
    dim = passages_embedding.shape[1]
    print('passage embedding shape: ' + str(passages_embedding.shape))
    
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passages_embedding)
    print("***** Done ANN Index *****")
    
    '''
    In passage_ Embedding for finding TOPK similarity in embedding
    '''
    # measure ANN mrr
    print("start searching ANN...")
    _, ANN_Index = cpu_index.search(query_embedding, args.ANN_topK)
    print("searching finish...")
    
    print("start generate neg data...")
    # Build hard negative based on the returned Ann index
    query_negative_passage = GenerateNegativePassaageID(ANN_Index,args.negative_num)
    if os.path.exists(neg_ann_data_path) == False:
        os.makedirs(neg_ann_data_path)
    # Organize new query_id, pos_pid ,neg_pid and save
    np.save(neg_ann_data, query_negative_passage)
    print("finished generating ann data")