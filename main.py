import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import math
import copy
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import tqdm
import re
def extract_last_number(text: str, keep_sign=True) -> str:
    """
    从文本中从后往前提取第一个数字（整数或小数）

    Args:
        text (str): 模型的回答
        keep_sign (bool): 是否保留正负号

    Returns:
        str: 提取到的数字（字符串形式），未找到则返回空字符串
    """
    # 反转字符串进行查找
    reversed_text = text[::-1]

    # 正则匹配数字（反转后匹配）
    if keep_sign:
        pattern = re.compile(r'((?:\d+(?:\.\d+)?|\.\d+)[-+]?)')  # 匹配整数、小数，带符号
    else:
        pattern = re.compile(r'(?:\d+(?:\.\d+)?|\.\d+)')
    matches = pattern.findall(reversed_text)
    
    if matches:
        # 取第一个匹配（注意要反转回来）
        num_reversed = matches[0]
        return num_reversed[::-1]
    
    return ""    
class my_embedding(nn.Module):
    def __init__(self,d_model,device):
        super(my_embedding,self).__init__()
        int2vec=torch.load("/home/yuanshixiang/shared/ysx/nlp/embedding_en.pth")
        self.int2vec={}
        for key,value in int2vec.items():
            self.int2vec[key]=value.to(device)
        self.linear=nn.Linear(300,d_model)
        self.d_model=d_model
    def forward(self,x):#bs s
        result=[]
        bs=len(x)
        sl=len(x[0])
        for s in x:
            for num in s:
                num=int(num.detach().cpu().item())
                result.append(self.int2vec[str(num)])
        result=torch.stack(result).reshape(bs,sl,300)
        return self.linear(result)*math.sqrt(self.d_model)
class my_generator(nn.Module):
    def __init__(self, d_model, vocab_size,device):
        super(my_generator, self).__init__()
        self.proj = nn.Linear(d_model, 300)
        self.linear1=nn.Linear(300,vocab_size)
        self.linear1.weight.data=torch.load("generator_matrix.pth")
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        for param in self.linear1.parameters():
            param.requires_grad=False
        self.linear2=nn.Linear(300,vocab_size)
        self.mask=torch.load("generator_mask.pth").to(device)#0 or 1
    def forward(self, x):
        x=self.proj(x)
        y1=self.linear1(x)
        y2=self.linear2(x)
        y=(1-self.mask)*y1+self.mask*y2
        return F.log_softmax(y,dim=-1)

class MyDataset(Dataset):
    def __init__(self, data_dir,mode='train'):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.train_src = []
        self.train_tgt = []
        with open(os.path.join(data_dir,"int2word_cn.json"), 'r', encoding='utf-8') as f:
            self.int2word_cn = json.load(f)
        with open(os.path.join(data_dir,"word2int_cn.json"), 'r', encoding='utf-8') as f:
            self.word2int_cn = json.load(f)
        with open(os.path.join(data_dir,"int2word_en.json"), 'r', encoding='utf-8') as f:
            self.int2word_en = json.load(f)
        with open(os.path.join(data_dir,"word2int_en.json"), 'r', encoding='utf-8') as f:
            self.word2int_en = json.load(f)
        # with open(os.path.join("int2word_en.json"), 'r', encoding='utf-8') as f:
        #     self.int2word_en = json.load(f)
        # with open(os.path.join("word2int_en.json"), 'r', encoding='utf-8') as f:
        #     self.word2int_en = json.load(f)
        self.src_vocab_size = len(self.word2int_en)
        self.tgt_vocab_size = len(self.word2int_cn)
        self.mode=mode
    def __len__(self):
        return len(self.train_src)
    def __getitem__(self, index):
        src = self.train_src[index]
        tgt = self.train_tgt[index]
        return src, tgt
    def build(self):
        if self.mode=='train':
            file_name = 'training.txt'
        elif self.mode=='test':
            file_name = 'testing.txt'
        else:
            file_name = 'validation.txt'
        with open(os.path.join(self.data_dir, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                src, tgt = line.strip().split('\t')
                src=[self.word2int_en[word] if word in self.word2int_en else self.unk_id for word in src.split()]
                tgt=[self.word2int_cn[word] if word in self.word2int_cn else self.unk_id for word in tgt.split()]
                self.train_src.append(src)
                self.train_tgt.append(tgt)

    def decode(self,id_list,mode="en"):     
        if mode=="en":
            s=[self.int2word_en[str(i)] for i in id_list]
        else:
            s=[self.int2word_cn[str(i)] for i in id_list]
        return s

class Attention(nn.Module):
    def __init__(self, d_model,n_heads,dropout=0.1):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, mask=None,key=None, value=None,):
        if key ==None:
            key=query
        if value==None:
            value=key
        batch_size = query.size(0)
        q,k,v=self.linears[0](query),self.linears[1](key),self.linears[2](value)
        q=q.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        k=k.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        v=v.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        attention_score=torch.matmul(q,k.transpose(-2,-1))/(self.d_k**0.5)
        if mask is not None:
            if len(mask.size())==3:
                mask=mask.unsqueeze(1)
            attention_score=attention_score.masked_fill(mask==0,-1e9)
        attention_weight=F.softmax(attention_score,dim=-1)
        attention_weight=self.dropout(attention_weight)
        weighted_value=torch.matmul(attention_weight,v)
        weighted_value=weighted_value.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_k)
        output=self.linears[3](weighted_value)
        return output
class FFN(nn.Module):
    def __init__(self, d_model, d_ff,dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_ff = d_ff
    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x
class SublayerConnection(nn.Module):
    def __init__(self, layer,dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer = layer
        self.d_model = layer.d_model
        self.norm = nn.LayerNorm(layer.d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,*args):
        args_list=list(args)
        original_x=args_list[0].clone()
        args_list[0]=self.norm(args_list[0])
        result=self.dropout(self.layer(*args_list))
        return original_x+result
class EncoderLayer(nn.Module):
    def __init__(self,attention,ffn,dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = SublayerConnection(attention,dropout)
        self.ffn = SublayerConnection(ffn,dropout)
        self.d_model = attention.d_model
    def forward(self, x, mask=None):
        attn_output=self.attention(x,mask)
        x=self.ffn(attn_output)
        return x
class Encoder(nn.Module):
    def __init__(self,encoder_layer,N):
        super(Encoder, self).__init__()
        self.N = N
        self.d_model = encoder_layer.d_model
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(N)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)
    def forward(self, x, mask=None):
        for i in range(self.N):
            x = self.layers[i](x,mask)
        x = self.norm(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,attention,ffn,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = SublayerConnection(attention,dropout)
        self.cross_attention = SublayerConnection(copy.deepcopy(attention),dropout)
        self.ffn = SublayerConnection(ffn,dropout)
        self.d_model = attention.d_model
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x=self.attention(x,tgt_mask)#mask用哪个关键看value
        x=self.cross_attention(x,src_mask,memory)
        x=self.ffn(x)
        return x
class Decoder(nn.Module):
    def __init__(self,decoder_layer,N,dropout=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.d_model = decoder_layer.d_model
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(N)])
        self.norm = nn.LayerNorm(decoder_layer.d_model)
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for i in range(self.N):
            x = self.layers[i](x,memory,src_mask,tgt_mask)
        x = self.norm(x)
        return x
class Embedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self, x):
        x = self.embedding(x)
        return x*math.sqrt(self.d_model)
#过每层神经网络，都要进行layer norm,然后进行dropout，然后再进行残差连接
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        for i in range(max_len):
            for j in range(d_model):
                if j%2==0:
                    pe[i,j]=math.sin(i/(10000**(j/d_model)))
                else:
                    pe[i,j]=math.cos(i/(10000**((j-1)/d_model)))
        self.register_buffer('pe',pe)
        self.pe.requires_grad_(False)
    def forward(self, x):#x : [batch_size,seq_len,d_model]
        x=x+self.pe[:x.size(1),:]
        return self.dropout(x)
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return F.log_softmax(self.proj(x),dim=-1)
class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,position,generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.position = position
        self.generator = generator
    def forward(self,src,tgt,src_mask,tgt_mask):
        src_embedding=self.src_embed(src)
        src_embedding=self.position(src_embedding)
        memory=self.encoder(src_embedding,src_mask)
        tgt_embedding=self.tgt_embed(tgt)
        tgt_embedding=self.position(tgt_embedding)
        output=self.decoder(tgt_embedding,memory,src_mask,tgt_mask)
        return self.generator(output)

def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,n_heads=8,dropout=0.1,checkpoint="",device=None,**kwargs):
    c=copy.deepcopy
    attn=Attention(d_model,n_heads,dropout)
    ffn=FFN(d_model,d_ff,dropout)
    position=PositionalEncoding(d_model,dropout)
    encoder_layer=EncoderLayer(c(attn),c(ffn),dropout)
    encoder=Encoder(encoder_layer,N)
    decoder_layer=DecoderLayer(c(attn),c(ffn),dropout)
    decoder=Decoder(decoder_layer,N)
    src_embed=Embedding(d_model,src_vocab)
    # src_embed=my_embedding(d_model,device)
    tgt_embed=Embedding(d_model,tgt_vocab)
    generator=Generator(d_model,tgt_vocab)
    # generator=my_generator(d_model,tgt_vocab,device)
    model=Transformer(encoder,decoder,src_embed,tgt_embed,position,generator)
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)
    if kwargs["resume"]:
        if os.path.exists(os.path.join(kwargs["exp_id"],checkpoint)):
            model.load_state_dict(torch.load(os.path.join(kwargs["exp_id"],checkpoint),map_location="cpu"))
        else: 
            model_ids=[int(float(extract_last_number(file_name))) for file_name in os.listdir(kwargs["exp_id"]) if file_name.endswith(".pth")]
            if len(model_ids)==0:
                print("Fail to load checkpoint")
                return model
            latest_model_id=max(model_ids)
            model_name="model_{}.pth".format(latest_model_id)
            model.load_state_dict(torch.load(os.path.join(kwargs["exp_id"],model_name),map_location="cpu"))
            print("Auto load {}".format(model_name))
        
    return model
def pad_seq(seq,pad_id,bos_id,eos_id,seq_len):
    seq=copy.deepcopy(seq)
    seq.insert(0,bos_id)
    seq.append(eos_id)
    if len(seq)<seq_len:
        seq.extend([pad_id for _ in range(seq_len-len(seq))])
    else:
        seq=seq[:seq_len]
    return seq
def mask_seq_pad(seq,pad_id,device):
    mask=(seq!=pad_id).to(torch.uint8).to(device).unsqueeze(-2)#pad is 0
    return mask
def mask_seq_future(seq_len,device):
    mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1)
    return (mask==0).to(torch.uint8).to(device)
def collate_fn_(x):
    return x
@torch.no_grad
def collate_fn(batch,pad_id,bos_id,eos_id,seq_len_limit=200,device=torch.device('cpu')):
    src,tgt=zip(*batch)#batch s
    max_src_len=max(len(s) for s in src)
    max_src_len=min(max_src_len+2,seq_len_limit)
    max_tgt_len=max(len(t) for t in tgt)
    max_tgt_len=min(max_tgt_len+2,seq_len_limit)
    src=[pad_seq(s,pad_id,bos_id,eos_id,max_src_len) for s in src]
    tgt=[pad_seq(t,pad_id,bos_id,eos_id,max_tgt_len) for t in tgt]
    src=torch.LongTensor(np.array(src)).to(device)
    tgt=torch.LongTensor(np.array(tgt)).to(device)
    tgt_y=tgt[:,1:]
    tgt=tgt[:,:-1]
    src_mask=mask_seq_pad(src,pad_id,device).to(device)#batch_size,1,seq_len        can see is 1
    tgt_mask=mask_seq_pad(tgt,pad_id,device).to(device)#batch_size,1,seq_len
    tgt_future_mask=mask_seq_future(tgt.size(-1),device).unsqueeze(0).repeat(tgt.size(0),1,1)#batch_size,seq_len,seq_len
    tgt_mask=tgt_mask&tgt_future_mask
    return src,tgt,src_mask,tgt_mask,tgt_y,(tgt_y!=pad_id).sum()
class LabelSmoothing:
    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    def __call__(self, x, target,norm):
        #x: (batch_size, seq_len, vocab_size)
        #target: (batch_size, seq_len)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(2,target.unsqueeze(-1),self.confidence)
        true_dist[:,:,self.padding_idx]=0.0
        mask=(target!=self.padding_idx).unsqueeze(-1).to(torch.float32)
        true_dist=true_dist*mask
        return self.criterion(x,true_dist.detach())/norm
class Logger:
    def __init__(self,log_dir='./',log_name="log.txt"):
        self.log_dir=log_dir
        self.log_name=log_name
        os.makedirs(log_dir,exist_ok=True)
    def log(self,msg):
        with open(os.path.join(self.log_dir,self.log_name),'a',encoding='utf-8') as f:
            f.write(msg+'\n')
def rate(step,model_size,factor,warmup):
    if step==0:
        step=1
    return factor*model_size**(-0.5)*min(step**(-0.5),step*warmup**(-1.5))

def run_epoch(model,data_loader,criterion,pad_id,bos_id,eos_id,seq_len_limit,device,optimizer=None,scheduler=None,mode='train',is_tqdm=False,logger=None,accum_iter=1):
    if mode=='train':
        model.train()
    else:
        model.eval()
    total_loss=0.0
    total_num_tokens=0
    data_iter=iter(data_loader)
    progress_bar=tqdm.tqdm(range(len(data_loader)),disable=not is_tqdm,leave=True)
    progress_bar.set_description(mode)
    for i in progress_bar:
        batch=next(data_iter)
        src, tgt, src_mask, tgt_mask ,tgt_y,ntokens=collate_fn(batch,pad_id,bos_id,eos_id,seq_len_limit,device)
        output = model(src, tgt, src_mask, tgt_mask)
        loss = criterion(output, tgt_y,ntokens)
        total_num_tokens+=ntokens
        total_loss += loss.item()*ntokens
        if mode=='train':
            loss.backward()
        if mode =="train" and i%accum_iter==0:
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        if logger != None and mode=="train":
            logger.log(f"loss: {loss.item():.4f}")
        if is_tqdm:
            if mode=="train":
                progress_bar.set_postfix(loss=f"{loss.item():.3f}",lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            else:
                progress_bar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss/total_num_tokens
def translate_test(data_dir,seq_len_limit,device,mode="translate",**kwargs):
    dataset=MyDataset(data_dir,mode='test')
    dataset.build()
    src_vocab_size=dataset.src_vocab_size
    tgt_vocab_size=dataset.tgt_vocab_size
    pad_id=dataset.pad_id
    bos_id=dataset.bos_id
    eos_id=dataset.eos_id
    model=make_model(src_vocab_size,tgt_vocab_size,device=device,**kwargs).to(device)
    src_list=[]
    tgt_list=[]
    prd_list=[]
    num_sentence=30
    if mode=="bleu":
        num_sentence=len(dataset)
    for i in range(num_sentence):
        if mode=="bleu":
            index=i
        else:
            index=random.randint(0,len(dataset)-1)
        src,tgt=dataset[index]
        # prediction=translate(model,src,pad_id,bos_id,eos_id,seq_len_limit,device)
        prediction=beam_search(model,src,pad_id,bos_id,eos_id,seq_len_limit,device)
        if prediction[-1]==eos_id:
            prediction.pop()
        src=dataset.decode(src,"en")
        tgt=dataset.decode(tgt,"cn")
        prediction=dataset.decode(prediction,"cn")
        src_list.append(src)
        tgt_list.append([tgt])  #计算bleu多个参考句子
        prd_list.append(prediction)
        if mode=="translate":
            print("*"*10,i,"*"*10)
            print("src:",''.join([word+' ' for word in src]))
            print("tgt:",''.join(tgt))
            print("predict:",''.join(prediction))

    return src_list,tgt_list,prd_list
def bleu_test(**kwargs):
    _,truth_list,predict_list=translate_test(mode="bleu",**kwargs)
    smoothing = SmoothingFunction().method4
    bleu_score = corpus_bleu(
        truth_list, 
        predict_list, 
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )
    return bleu_score
def translate(model,src,pad_id,bos_id,eos_id,seq_limit,device):
    tgt=[]
    for i in range(seq_limit):
        batch=[(src,tgt)]
        src_, tgt_, src_mask_, tgt_mask_ ,tgt_y_,ntokens_=collate_fn(batch,pad_id,bos_id,eos_id,seq_limit,device)
        output=model(src_,tgt_,src_mask_,tgt_mask_)#bs seq vocob
        prediction=torch.argmax(output[0,-1,:],dim=-1)
        tgt.append(prediction.cpu().item())
        if prediction==eos_id:
            break
    return tgt
def beam_search(model,src,pad_id,bos_id,eos_id,seq_limit,device):
    k=3
    thresh=-0.8
    tgts=[[] for i in range(k)]
    scores=torch.zeros(k).to(device).unsqueeze(-1)
    lens=torch.ones(k).to(device).unsqueeze(-1)
    lives=torch.ones(k).to(device).unsqueeze(-1)
    for i in range(seq_limit):
        batch=[(src,tgt) for j,tgt in enumerate(tgts)]
        src_, tgt_, src_mask_, tgt_mask_ ,tgt_y_,ntokens_=collate_fn(batch,pad_id,bos_id,eos_id,seq_limit,device)
        output=model(src_,tgt_,src_mask_,tgt_mask_)#bs seq vocob
        next_logit=output[:,-1,:]
        bs,vocab_size=next_logit.shape
        new_lens=lens+lives
        cand_scores=(scores*lens+next_logit*lives)/new_lens
        rank_scores=cand_scores*new_lens/new_lens**(-3)
        rank_scores=rank_scores.view(-1)
        _,indexes=rank_scores.topk(k)
        mask=next_logit.flatten()[indexes]>thresh
        legal_topk=mask.to(torch.int).sum().cpu().item()
        if legal_topk==0:
            legal_topk=1
        indexes=indexes[:legal_topk]
        new_scores=cand_scores.view(-1)[indexes]
        new_tgts=[]
        new_lives=torch.ones(len(indexes)).to(device)
        lens=torch.ones(len(indexes)).to(device)
        for j,index in enumerate(indexes):
            original_id=index//vocab_size
            word_id=(index%vocab_size).cpu().item()
            new_tgts.append(tgts[original_id])
            new_tgts[-1].append(word_id)
            new_lives[j]=lives[original_id][0]
            if word_id==eos_id:
                new_lives[j]=0
            lens[j]=new_lens[original_id][0]
        scores=new_scores.unsqueeze(-1)
        tgts=new_tgts
        lens=lens.unsqueeze(-1)
        lives=new_lives.unsqueeze(-1)
        if lives.sum()==0:
            break
    best_idx=torch.argmax(scores.squeeze(-1))
    best_tgt=tgts[best_idx]
    wash_tgt=[]
    for i,word_id in enumerate(best_tgt):
        wash_tgt.append(word_id)
        if word_id==eos_id:
            break
    return wash_tgt
def train_worker(rank,config):
    gpu_id=rank
    is_master=rank==0
    torch.cuda.set_device(gpu_id)
    world_size=config["world_size"]
    dist.init_process_group(backend="nccl",init_method='env://',rank=rank,world_size=world_size)
    device = torch.device(f"cuda:{gpu_id}")
    config["device"]=device
    exp_id=config["exp_id"]
    if is_master:
        os.makedirs(config["exp_id"],exist_ok=True)
    logger=None if not is_master else Logger(log_dir=exp_id)
    data_dir=config["data_dir"]
    dataset=MyDataset(data_dir,mode='train')
    dataset.build()
    val_dataset=MyDataset(data_dir,mode='valid')
    val_dataset.build()
    pad_id=dataset.pad_id
    bos_id=dataset.bos_id
    eos_id=dataset.eos_id
    src_vocab_size=dataset.src_vocab_size
    tgt_vocab_size=dataset.tgt_vocab_size
    lr=config["lr"]
    seq_len_limit=config["seq_len_limit"]

    model=make_model(src_vocab_size,tgt_vocab_size,**config).to(device)
    model=DDP(model,device_ids=[gpu_id])

    criterion=LabelSmoothing(tgt_vocab_size,pad_id,smoothing=config["label_smoothing"])
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.98),eps=1e-9)
    scheduler=None
    if config["use_scheduler"]:
        scheduler=LambdaLR(optimizer,lr_lambda=lambda step: rate(step,config["d_model"],1,warmup=config["warmup"]) )
    batch_size=config["batch_size"]//world_size
    num_workers = 0
    sampler=DistributedSampler(dataset,shuffle=True)
    train_loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=num_workers,collate_fn=collate_fn_)#ensure it will not try to transfer data to tensor ,which will cause problem
    val_sampler=DistributedSampler(val_dataset,shuffle=False)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,sampler=val_sampler,num_workers=num_workers,collate_fn=collate_fn_)
    for epoch in range(config["num_epochs"]):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        train_loss=run_epoch(model,train_loader,criterion,pad_id,bos_id,eos_id,seq_len_limit,device,optimizer,scheduler,mode='train',is_tqdm=is_master,accum_iter=config["accum_iter"])
        if is_master:
            tqdm.tqdm.write(f"Epoch {epoch}, Mean Loss {train_loss}")
            logger.log(f"Epoch {epoch}, Mean Loss {train_loss}")
        if epoch%config["eval_every"]==0:
            val_loss=run_epoch(model,val_loader,criterion,pad_id,bos_id,eos_id,seq_len_limit,device,mode='validation')
            if is_master:
                tqdm.tqdm.write(f"Validation Mean Loss {val_loss}")
                logger.log(f"Validation Mean Loss {val_loss}")
        if epoch%config["save_every"]==0 and is_master:
            torch.save(model.module.state_dict(),os.path.join(exp_id,f"model_{epoch}.pth"))
        if epoch%config["bleu_test_every"]==0:
            bleu_score=bleu_test(**config)
            tqdm.tqdm.write(f"Bleu Score {bleu_score}")
            logger.log(f"Bleu Score {bleu_score}")
def distributed_train(config):
    world_size=config["world_size"]
    os.environ['MASTER_ADDR']= 'localhost'
    os.environ['MASTER_PORT'] = '12509'
    mp.spawn(train_worker,nprocs=world_size,args=(config,))

if __name__=="__main__":
    random.seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    config={
            "d_model":512,
            "d_ff":2048,
            "n_heads":8,
            "N":6,
            "dropout":0.1,
            "seq_len_limit":72,
            "accum_iter":1,
            "lr":0.0002,        #real lr= accum_iter*lr
            "batch_size":160, #real total batch_size =accum_iter*batch_size
            "label_smoothing": 0.1,
            "use_scheduler": False,
            "warmup":1000,
            "num_epochs":61,
            "eval_every":1,
            "bleu_test_every":5,
            "save_every":1,
            "data_dir":"/home/yuanshixiang/shared/data/cmn-eng-simple",
            "world_size":1,
            "device":"cuda:0",              #use for test bleu translate
            "checkpoint":"model_60.pth",
            "resume":1,                      #
            "exp_id":"2",
            }
    # distributed_train(config)
    translate_test(**config)
    # print(bleu_test(**config))
    #test 100 0.2000   50 val 0.2018
    #0
    #1 rl 
