import datasets
import torch
torch.multiprocessing.set_start_method('spawn')
import sys
import os
import argparse
from tqdm import tqdm
import random
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve,classification_report
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import WEIGHTS_NAME, CONFIG_NAME


random_seed = 2022
random.seed(random_seed)
torch.manual_seed(random_seed)

parser = argparse.ArgumentParser(description='Pre training model configuration')
parser.add_argument('--model', default='electra',help='pre train model type')
parser.add_argument('--model_size', default='Small',help='pre train model size')
# dataset ：【imdb、yelp2、sst5、yelp5、mr、custom、】（custom Custom data file path）
# The custom dataset must contain (text key, label key), and must contain the train and valid files
parser.add_argument('--dataset', default='custom',help='pre train dataset')
parser.add_argument('--inference', default="valid",help='Choose whether to use a test set or a validation set')
parser.add_argument('--gpu_num', type=int ,default=4,help='pre train gpu num')
parser.add_argument('--data_path', default='./datasets/',help='the save dir of dataset')
parser.add_argument('--save_model', default='./save_models/',help='the save dir of fine-tunning model path')
parser.add_argument('--pretrain_model', default='./pretrain_model/',help='the save dir of original model path')
parser.add_argument('--batch_size',type=int ,default=32,help='the batch size of training process')
parser.add_argument('--max_epoch', type=int,default=10,help='the max epoch of training process')
parser.add_argument('--lr', default = 2e-5,help='the learning rate of fine-tunning process')
parser.add_argument('--eps', default = 1e-8,help='the eps of fine-tunning process')
parser.add_argument('--max_length', type=int,default = 512,help='the max_length of input sequence')
parser.add_argument("--local_rank", type=int,default=-1, help="local rank")
parser.add_argument("--rank", type=int,default=-1, help="rank")
parser.add_argument("--result_path", default = './result/', help="result path")
parser.add_argument("--loadmodel", type=bool, default = False, help="isload model")
parser.add_argument("--loadmodelpath", type=str, default='ance7word5',help="load model path")
parser.add_argument('--jsondata', default='./jsondata/custom/',help='the save dir of custom json dataset（if you use custom data）')
parser.add_argument("--issavemodel", type=bool, default = False, help="is save model?")
parser.add_argument("--fewshot", type=bool, default = False, help="is use few shot data")
parser.add_argument("--fewshotnum", type=int, default = 8, help="few shot data size")
parser.add_argument("--zeroshot", type=bool, default = False, help="is use zero shot?")
parser.add_argument("--fewshotdatapath", type=str, default = './fewdata/', help="few shot data path")
parser.add_argument("--num_class", type=int, default = 2, help="is classify num labels")
args = parser.parse_args()

torch.cuda.current_device()
# Global definition to determine whether multi machine multi GPU or single machine multi GPU
if args.gpu_num>1 and args.gpu_num<=8:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
elif args.gpu_num>8:
    dist.init_process_group(backend='nccl')
    rank=torch.distributed.get_rank()
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_epoch = args.max_epoch
lr = args.lr
if args.model_size=='large':
    lr=1e-5
eps = args.eps
max_length = args.max_length
batch_size = args.batch_size
dataset_path = args.data_path + args.dataset

if args.loadmodel:
    loadmodelpath=args.pretrain_model+args.loadmodelpath
    print("loadmodelpath:",loadmodelpath)
else:
    print("--loadmodel=false")
    sys.exit()
print("lr:",lr)


def train(model, dataset, test_loader, max_epoch):

    # model.to(device)
    if  args.gpu_num>1 and args.gpu_num<=8:
        sampler = DistributedSampler(dataset['train'])
        train_loader = DataLoader(dataset['train'], batch_size=batch_size,sampler=sampler)
    elif args.gpu_num>8:
        sampler = DistributedSampler(dataset['train'],num_replicas=world_size, rank=rank)
        train_loader = DataLoader(dataset['train'], batch_size=batch_size,sampler=sampler)
    else:
        train_loader = DataLoader(dataset['train'], batch_size=batch_size,shuffle=True)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr,eps = eps)
    total_steps = len(train_loader) * max_epoch*args.gpu_num
    warmup_steps=0
    if args.model_size=='large':
        warmup_steps=0.1*len(train_loader)/args.gpu_num
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = warmup_steps, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    model.train()

    for epoch in range(max_epoch):
        # dataset = dataset.shuffle()
        # train_loader = DataLoader(dataset['train'], batch_size=batch_size)
        if args.gpu_num>1:
            sampler.set_epoch(epoch)
        total_train_loss = 0
        for iter_num, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.forward(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            # clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if iter_num % 20 == 0:
                print("epoth: %d, iter_num: %d, loss: %.4f" % (epoch, iter_num, loss.item()))

        if not args.loadmodel:
            if args.gpu_num>1:
                output_dir=args.save_model+args.model+'-'+args.model_size+'_'+args.dataset+'_epoch_'+str(epoch)+'_multgpu'+str(args.gpu_num)
            else:
                output_dir=args.save_model+args.model+'-'+args.model_size+'_'+args.dataset+'_epoch_'+str(epoch)
        else:
            output_dir=args.save_model+args.loadmodelpath+'_'+args.dataset+'_epoch_'+str(epoch)+'trans'

        if args.gpu_num>1:
            if dist.get_rank() == 0:
                validation(model,test_loader)
        else:
            validation(model,test_loader)
        if args.issavemodel:
            if args.gpu_num>1:
                if dist.get_rank() == 0:
                    save_model(model,output_dir)
            else:
                save_model(model,output_dir)
        print("Epoch: %d, Average training loss: %.4f" %(epoch, total_train_loss/len(train_loader)))


def validation(model,test_loader):
    model.eval()
    y_true=[]
    y_pred=[]
    total_eval_loss=0

    for batch in test_loader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.forward(**batch)

        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()
        prediction = torch.argmax(logits, 1)
        prediction = prediction.detach().cpu().tolist()
        label_ids = batch['labels'].to('cpu').tolist()
        # print(prediction)
        # print(label_ids)
        y_pred+=prediction
        y_true+=label_ids
    if args.num_class==2:
        target_names = ['Negative', 'Positive']
    elif args.num_class==5:
        target_names = ['Very Negative','Negative','Neutral', 'Positive','Very Positive']
    else:
        sys.exit()
    print(classification_report(y_true, y_pred, target_names=target_names,digits=4))
    if os.path.exists(args.result_path) == False:
        os.makedirs(args.result_path)
    if not args.loadmodel:
        if args.zeroshot:
            file = open(args.result_path + args.model+'-'+args.model_size+ "_" + args.dataset+'_zeroshot_result.txt',"a+")
        elif args.fewshot:
            file = open(args.result_path + args.model+'-'+args.model_size+ "_" + args.dataset+'_fewshot'+str(args.fewshotnum)+'_result.txt',"a+")
        else:
            file = open(args.result_path + args.model+'-'+args.model_size+ "_" + args.dataset + "_result.txt", "a+")
    else:
        if args.zeroshot:
            file = open(args.result_path + args.loadmodelpath+ "_" + args.dataset+'_zeroshot_result.txt',"a+")
        elif args.fewshot:
            file = open(args.result_path + args.loadmodelpath+ "_" + args.dataset+'_fewshot'+str(args.fewshotnum)+'_result.txt',"a+")
        else:
            file = open(args.result_path + args.loadmodelpath+ "_" + args.dataset+'trans' + "_result.txt", "a+")

    file.write(classification_report(y_true, y_pred, target_names=target_names,digits=4) + '\n' + '\n')
    file.close()
    print("Average testing loss: %.4f"%(total_eval_loss/len(test_loader)))
    print("-------------------------------")

def save_model(model,output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    print("save model in :"+output_dir)


if __name__ == "__main__":
    #model
    print("classify class is "+str(args.num_class)+" and use load model")
    model_path = os.path.join(args.pretrain_model, args.model)
    save_path = loadmodelpath
    
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=args.num_class)

    #data
    if args.fewshot or args.zeroshot:
        #few shot and zero shot finetune
        dataset_path=args.fewshotdatapath+args.dataset
        if args.fewshot:
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
            dataset = {}
            train_path="/train"+str(args.fewshotnum)+".json"
            dataset['train'] = datasets.dataset_dict.DatasetDict.from_json(dataset_path + train_path)
            if args.inference == "test":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(dataset_path+ "test.jsonl")
            elif args.inference == "valid":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(dataset_path + "valid.jsonl")
            else:
                print("inference must be test or valid")
                exit(0)
            print("format " + args.dataset + " dataset....")
            dataset['train'] = dataset['train'].map(encode, batched=True)
            dataset['test'] = dataset['test'].map(encode, batched=True)
            dataset['train'] = dataset['train'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['test'] = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['train'].set_format(type='torch',
                                        columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                 'labels'])
            dataset['test'].set_format(type='torch',
                                       columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                'labels'])
        elif args.zeroshot:
            print("zero shot inference")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
            dataset = {}
            if args.inference == "test":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(dataset_path + "test.jsonl")
            elif args.inference == "valid":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(dataset_path + "valid.jsonl")
            else:
                print("inference must be test or valid")
                exit(0)
            print("format " + args.dataset + " dataset....")
            dataset['test'] = dataset['test'].map(encode, batched=True)
            dataset['test'] = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['test'].set_format(type='torch',
                                       columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                'labels'])
    else:
    # Fintune loading and processing datasets
        if args.dataset == 'imdb':
            print("load imdb dataset...")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

            if os.path.exists(dataset_path) == False:
                dataset = datasets.load_dataset('imdb', cache_dir='./datasets')
                dataset.save_to_disk(dataset_path)
            else:
                print(dataset_path)
                dataset = datasets.load_from_disk(dataset_path)
            print(dataset_path)
            print("format " + args.dataset + " dataset....")
            dataset = dataset.map(encode, batched=True)
            dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        elif args.dataset == 'yelp2':
            print("load yelp2 dataset...")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length',max_length=max_length)

            if os.path.exists(dataset_path) == False:
                dataset = datasets.load_dataset('yelp_polarity', cache_dir='./datasets')
                dataset.save_to_disk(dataset_path)
            else:
                dataset = datasets.load_from_disk(dataset_path)

            print("format " + args.dataset + " dataset....")
            dataset = dataset.map(encode, batched=True)
            dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        elif args.dataset == 'custom':
            #you can put the data in jsondata path in train.json and test.json to finetunning
            # path in args.jsondata
            print("Load custom datasets...")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

            dataset = {}
            if args.inference == "test":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(args.jsondata + "test.jsonl")
            elif args.inference == "valid":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(args.jsondata + "valid.jsonl")
            else:
                print("inference must be test or valid")
                exit(0)
            print("format " + args.dataset + " dataset....")
            dataset['train'] = dataset['train'].map(encode, batched=True)
            dataset['test'] = dataset['test'].map(encode, batched=True)
            dataset['train'] = dataset['train'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['test'] = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['train'].set_format(type='torch',
                                                columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                            'labels'])
            dataset['test'].set_format(type='torch',
                                                columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                            'labels'])
            print(len(dataset['train']))
        elif args.dataset == 'mr':
            print("Load mr datasets...")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
            jsondata_path = './jsondata/MR/'
            dataset = {}
            dataset['train'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path+ "train.json")
            if args.inference == "test":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path + "test.jsonl")
            elif args.inference == "valid":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path + "valid.jsonl")
            else:
                print("inference must be test or valid")
                exit(0)
            print("format " + args.dataset + " dataset....")
            dataset['train'] = dataset['train'].map(encode, batched=True)
            dataset['test'] = dataset['test'].map(encode, batched=True)
            dataset['train'] = dataset['train'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['test'] = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['train'].set_format(type='torch',
                                                columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                            'labels'])
            dataset['test'].set_format(type='torch',
                                                columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                            'labels'])
            print(len(dataset['train']))

        elif args.dataset == 'sst5':
            print("Load sst5 datasets...")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
            jsondata_path = './jsondata/sst5/'
            dataset = {}
            dataset['train'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path+ "train.jsonl")
            if args.inference == "test":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path + "test.jsonl")
            elif args.inference == "valid":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path + "valid.jsonl")
            else:
                print("inference must be test or valid")
                exit(0)
            print("format " + args.dataset + " dataset....")
            dataset['train'] = dataset['train'].map(encode, batched=True)
            dataset['test'] = dataset['test'].map(encode, batched=True)
            dataset['train'] = dataset['train'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['test'] = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['train'].set_format(type='torch',
                                        columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                 'labels'])
            dataset['test'].set_format(type='torch',
                                       columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                'labels'])
        elif args.dataset =='yelp5':
            print("load yelp5 dataset...")
            print("Load sst5 datasets...")
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
            jsondata_path = './jsondata/yelp5/'
            dataset = {}
            dataset['train'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path+ "train.jsonl")
            if args.inference == "test":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path+ "test.jsonl")
            elif args.inference == "valid":
                dataset['test'] = datasets.dataset_dict.DatasetDict.from_json(jsondata_path + "valid.jsonl")
            else:
                print("inference must be test or valid")
                exit(0)
            print("format " + args.dataset + " dataset....")
            dataset['train'] = dataset['train'].map(encode, batched=True)
            dataset['test'] = dataset['test'].map(encode, batched=True)
            dataset['train'] = dataset['train'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['test'] = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)
            dataset['train'].set_format(type='torch',
                                        columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                 'labels'])
            dataset['test'].set_format(type='torch',
                                       columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                'labels'])

        else:
            print("no dataset name " + args.dataset)
            sys.exit()

    test_loader = DataLoader(dataset['test'], batch_size=batch_size)
    # if args.inference == "test":
    #     test_loader = DataLoader(dataset['test'], batch_size=batch_size)
    # elif args.inference == "valid":
    #     test_loader = DataLoader(dataset['valid'], batch_size=batch_size)
    # else:
    #     print("inference must be test or valid")
    #     exit(0)

    model.to(device)
    #model
    if  args.gpu_num>1:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)
    if not args.zeroshot:
        if not args.fewshot:
        # train model
            print("finetune " + args.model + " model...")
            train(model,dataset,test_loader,max_epoch)
        else:
            print("few shot train " + args.model + " model...")

            train(model,dataset,test_loader,max_epoch)
    else:
        print("zero shot inference " + args.model + " model...")
        validation(model,test_loader)
