import random
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import init
import torch.nn.functional as Func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class Recurrent_Paragraph_Generative_Encoder(nn.Module):
    def __init__(self,embeding_size,vocab_size,seq_length,sen_enc_conv1d_out=1024):
        super(Recurrent_Paragraph_Generative_Encoder,self).__init__()
        self.word_embeding=nn.Embedding(vocab_size,embeding_size)
        self.pool_size=seq_length
        self.cnn1=nn.Conv1d(embeding_size,sen_enc_conv1d_out,kernel_size=3,stride=1)
        self.maxpooling1=nn.MaxPool1d(kernel_size=seq_length-2)
        self.cnn2=nn.Conv1d(sen_enc_conv1d_out,sen_enc_conv1d_out,kernel_size=3,stride=1)
        self.maxpooling2=nn.MaxPool1d(kernel_size=seq_length-4)
        self.cnn3=nn.Conv1d(sen_enc_conv1d_out,sen_enc_conv1d_out,kernel_size=3,stride=1)
        self.maxpooling3=nn.MaxPool1d(kernel_size=seq_length-6)
        self.vocabsize=vocab_size
    def forward(self,pre_sentence): #把某一个decoder生成的前一个句子输入cnn
        sentence_embeding=self.word_embeding(pre_sentence)
        sentence_embeding_transposed=sentence_embeding.transpose(1,2)
        cnn_out1=self.cnn1(sentence_embeding_transposed)
        cnn_feats1=self.maxpooling1(cnn_out1).squeeze()
        cnn_out2=self.cnn2(cnn_out1)
        cnn_feats2=self.maxpooling2(cnn_out2).squeeze()
        cnn_out3=self.cnn3(cnn_out2)
        cnn_feats3=self.maxpooling3(cnn_out3).squeeze()
        sen_feats=torch.cat((cnn_feats1,cnn_feats2,cnn_feats3),dim=-1)
        return sen_feats

class Recurrent_Paragraph_Generative_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, decoder_num_layers, sen_enco_num_layers,num_global_features, num_regions, num_conv1d_out=1024, teach_forcing_rate=1.0, max_seq_length=15,max_sentence_num=7, dropout_rate=0):
        super(Recurrent_Paragraph_Generative_Decoder,self).__init__()
        self.decoder_num_layers = decoder_num_layers
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        self.max_sentence_num = max_sentence_num
        self.vocab_size = vocab_size
        self.num_regions = num_regions
        self.num_conv1d_out = num_conv1d_out
        self.teach_forcing_rate = teach_forcing_rate

        self.embed_h = nn.Linear(num_global_features + num_conv1d_out * sen_enco_num_layers,
                                 hidden_size * decoder_num_layers)
        self.embed_c = nn.Linear(num_global_features + num_conv1d_out * sen_enco_num_layers,
                                 hidden_size * decoder_num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, decoder_num_layers, batch_first=True, dropout=dropout_rate)
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence_encoder = Recurrent_Paragraph_Generative_Encoder(embed_size, vocab_size, max_seq_length, num_conv1d_out)

    def sampler(self, global_features, topic_vector, max_single_sen_len, max_sen_num, ini_decoder_state=None):
        """"Generate findings in the testing process"""

        last_input, last_state = self._combine_vis_text(global_features, topic_vector)
        # The dimension of predicted_findings is denoted at the end
        predicted_findings = []
        # sentence recurrent loop
        for num_sen in range(max_sen_num):
            # predicted sentences (indices), used for generating the preceding topic vector
            predicted_single_sentence = []

            # word recurrent loop
            for time_step in range(max_single_sen_len):
                decoder_output_t, decoder_state_t = self._word_step(last_input, last_state)
                last_state = decoder_state_t
                _, pre_indices = decoder_output_t.max(dim=-1)
                pre_indices = pre_indices.unsqueeze(1)
                predicted_single_sentence.append(pre_indices)
                last_input = self.word_embed(pre_indices)

            predicted_single_sentence = torch.cat(predicted_single_sentence, dim=1)
            sen_vector = self.sentence_encoder(predicted_single_sentence)
            last_input, last_state = self._combine_vis_text(global_features, sen_vector)
            predicted_single_sentence = predicted_single_sentence.unsqueeze(-1)
            predicted_findings.append(predicted_single_sentence)

        predicted_findings = torch.cat(predicted_findings, dim=2)
        predicted_findings = predicted_findings.transpose(1, 2)

        return predicted_findings
    def _forward_step(self, input_t, state_t):#逐词生成句子
        output_t, state_t = self.lstm(input_t, state_t)
        out_t_squ = output_t.squeeze(dim=1)
        out_fc = Func.log_softmax(self.linear(out_t_squ), dim=-1)
        return out_fc, state_t

    def _combine_vis_text(self, global_features, sen_vec):
        ini_input = torch.zeros(global_features.shape[0]).long().to(device)
        last_input = self.word_embed(ini_input).unsqueeze(1)
        con_features = torch.cat((global_features, sen_vec), dim=1)
        h_stat = self.embed_h(con_features)
        h_stat = h_stat.view((h_stat.shape[0], self.decoder_num_layers, -1)).transpose(0, 1).contiguous()
        c_stat = self.embed_c(con_features)
        c_stat = c_stat.view((c_stat.shape[0], self.decoder_num_layers, -1)).transpose(0, 1).contiguous()
        last_state = (h_stat, c_stat)
        return last_input, last_state

    def forward(self, global_features, topic_vector, findings, fin_lengths):#生成整个报告
        gt_packed, decoder_outputs_packed = None, None
        last_input, last_state = self._combine_vis_text(global_features, topic_vector)
        for num_sen in range(findings.shape[1]):#循环每个句子

            fin_sen_embedded = self.word_embed(findings[:, num_sen, :])#embeding ground truth
            decoder_input = torch.cat((last_input, fin_sen_embedded), dim=1)
            # The num of words for each sentence in the finding
            fin_sen_lengths = fin_lengths[:, num_sen]

            # pack groundtruth
            gt_fin_sen_packed = pack_padded_sequence(findings[:, num_sen, :], fin_sen_lengths, batch_first=True, enforce_sorted=False)[0]
            if num_sen == 0:
                gt_packed = gt_fin_sen_packed
            else:
                gt_packed = torch.cat((gt_packed, gt_fin_sen_packed), dim=0)

            fin_sen_packed = pack_padded_sequence(decoder_input, fin_sen_lengths, batch_first=True,
                                                  enforce_sorted=False)
            out_lstm, _ = self.lstm(fin_sen_packed, last_state)
            padded_outs, _ = pad_packed_sequence(out_lstm, batch_first=True)
            fin_sen_outputs = Func.log_softmax(self.dropout(self.linear(padded_outs)), dim=-1)

            fin_sen_outputs_packed = pack_padded_sequence(fin_sen_outputs, fin_sen_lengths, batch_first=True, enforce_sorted=False)[0]
            if num_sen == 0:
                decoder_outputs_packed = fin_sen_outputs_packed
            else:
                decoder_outputs_packed = torch.cat((decoder_outputs_packed, fin_sen_outputs_packed), dim=0)

            _, predicted_sentences = fin_sen_outputs.max(dim=-1)
            if random.random() < self.teach_forcing_rate:
                sen_vector = self.sentence_encoder(findings[:, num_sen, :])
            else:
                sen_vector = self.sentence_encoder(predicted_sentences)
            last_input, last_state = self._combine_vis_text(global_features, sen_vector)

        return gt_packed, decoder_outputs_packed
