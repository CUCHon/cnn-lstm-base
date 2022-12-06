import random
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import init
import torch.nn.functional as Func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from Model.RPGmodels import Recurrent_Paragraph_Generative_Encoder

class Sentence_Generative_Model_Decoder(nn.Module):
    def __init__(self,embeding_size,hidden_size,vocab_size,num_layers,num_global_features,num_conv1d_out=1024,teach_forcing_rate=0.5,max_sequnece=15,dropout_rate=0.01):

        super(Sentence_Generative_Model_Decoder,self).__init__()
        self.embeding_size=embeding_size
        self.vocab_size=vocab_size
        self.teach_forcing_rate=teach_forcing_rate
        self.max_sequence=max_sequnece
        self.num_conv1d_out=num_conv1d_out
        self.vis_embed=nn.Linear(num_global_features,embeding_size)#这里把图像特征提到embeding-size的向量里
        self.word_embed= nn.Embedding(vocab_size,embeding_size)#这里把word embedding
        self.dropout= nn.Dropout(dropout_rate)
        self.lstm= nn.LSTM(embeding_size,hidden_size,num_layers,batch_first=True,dropout=dropout_rate)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.sentence_encoder= Recurrent_Paragraph_Generative_Encoder(embeding_size,vocab_size,max_sequnece,num_conv1d_out)
    def forward(self,global_features,impressions,imp_lengths):

        vis_embedings=self.vis_embed(global_features)
        ini_input=vis_embedings.unsqueeze(1)
        impression_embeding=self.word_embed(impressions)
        decoder_input=torch.cat((ini_input,impression_embeding),1)# concat起来图和impression的向量
        imp_packed=pack_padded_sequence(decoder_input,imp_lengths,batch_first=True,enforce_sorted=False)#进入rnn之前要pack
        lstm_out, =self.lstm(imp_packed)
        padded_out=pad_packed_sequence(lstm_out,batch_first=True)#rnn出来再pad
        decoder_output=Func.log_softmax(self.dropout(self.linear(padded_out)),dim=-1)#计算输出概率
        decoder_output_packed=pack_padded_sequence(decoder_output,imp_lengths,batch_first=True,enforce_sorted=False)[0]
        gt_packed=pack_padded_sequence(impressions,imp_lengths,batch_first=True,enforce_sorted=False)[0]
        __,predict_sntences=decoder_output.max(dim=-1)
        if self.teach_forcing_rate > random.Random():
            topic_vec=self.sentence_encoder(predict_sntences)
        else:
            topic_vec=self.sentence_encoder(impressions)
        return gt_packed,decoder_output_packed,topic_vec
    def sampler(self,global_features,max_len,ini_decoder_state=None):#采样策略
        vis_embedings = self.vis_embed(global_features)
        ini_input=vis_embedings.unsqueeze(1)
        decoder_input_t=ini_input
        decoder_state_t=ini_decoder_state
        impression_ids=[]
        for i in range(max_len):
            decoder_output_t,decoder_state_t=self._forward_step(decoder_input_t,decoder_state_t)
            pre_values,pre_indices=decoder_output_t.max(dim=-1)
            pre_indices=pre_indices.unsqueeze(1)
            impression_ids.append(pre_indices)
            decoder_input_t=self.word_embed(pre_indices)
        impression_ids=torch.cat(impression_ids,dim=1)
        topic_vec=self.sentence_encoder(impression_ids)
        return impression_ids,topic_vec #每次生成一句话
    def _forward_step(self,input_t,state_t):#lstm中每一步生成一个词
        output_t,state_t=self.lstm(input_t,state_t)
        out_t_squ=output_t.squeeze(dim=1)
        out_fc=Func.log_softmax(self.linear(out_t_squ),dim=-1)
        return out_fc,state_t
