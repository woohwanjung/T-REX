from collections import OrderedDict

import torch
from torch import nn
from pytorch_transformers import *
from torch.nn.utils.rnn import pad_sequence

from settings import conditional_profiler, DEBUG_DIMENSION_MATCHING


def map2vec(context, mapping):
    res = torch.bmm(mapping, context)
    return res


class AttentionModel(nn.Module):
    OPT_ADDITIVE = 1
    OPT_MULTIPLICATIVE = 2

    def __init__(self, context_size, state_size, hidden_size, att_opt=OPT_ADDITIVE):
        super(AttentionModel, self).__init__()

        self.net_W = nn.Linear(state_size, hidden_size)
        self.net_U = nn.Linear(context_size, hidden_size)
        self.net_v = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

        self.attn_softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def forward(self, context, state):
        '''
        :param context: (batchsize, state_length, state_size)
        :param state:   (batchsize, context_length, context_size)
        :return: output_context, att_weight
                output_context (batchsize, state_length, context_size)
                att_weight (batchsize, state_length, context_length)
        '''

        batchsize, state_length, state_size = state.shape
        _batchsize, context_length, context_size = context.shape
        hidden_size = self.hidden_size
        assert batchsize == _batchsize

        v_state = self.net_W(state)  # (batchsize, state_length, hidden_size)
        v_context = self.net_U(context)  # (batchsize, context_length, hidden_size)

        att_weight = []
        output_context = []
        for s in range(state_length):
            vs = v_state[:, s]  # (batchsize, hidden)
            att_weight_s = vs.view(batchsize, 1, hidden_size) + v_context  # (batchsize, context_length, hidden_size)
            att_weight_s = self.tanh(att_weight_s)
            att_weight_s = self.net_v(att_weight_s)  # (batchsize, context_length, 1)
            att_weight_s = self.attn_softmax(att_weight_s)

            output_s = torch.bmm(att_weight_s.transpose(1, 2), context)

            att_weight.append(att_weight_s.squeeze(-1))
            output_context.append(output_s.squeeze(1))

        att_weight = torch.stack(att_weight, dim=1)
        output_context = torch.stack(output_context, dim=1)

        return output_context, att_weight


class AttentionModelLargeMemory(nn.Module):

    def __init__(self, context_size, query_size, hidden_size):
        super().__init__()
        # Additive attention
        if DEBUG_DIMENSION_MATCHING:
            hidden_size -= 1
        self.linear_c = torch.nn.Linear(context_size, hidden_size)
        self.linear_q = torch.nn.Linear(query_size, hidden_size, bias=False)
        self.linear_o = torch.nn.Linear(hidden_size, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        # Multihead
        # num_head = 4
        # self.attention = nn.MultiheadAttention(vect_size, num_head)

    def forward(self, query, context, mapping_mask):
        # print(query.shape)
        # print(context.shape)
        # print(mapping_mask.shape)

        n_batches, h_input = query.shape
        res_q = self.linear_q(query)  # B*H
        res_c = self.linear_c(context)  # B*M*H

        epsilon = 0.00001

        # B*M*H
        tmp = res_q.unsqueeze(1) + res_c
        tmp = self.tanh(tmp)
        weights_logit = self.linear_o(tmp)
        weights_exp = mapping_mask * torch.exp(weights_logit)
        weights = weights_exp / (weights_exp.sum(dim=1, keepdim=True) + epsilon)
        #weights = self.softmax(weights_logit)

        #output = torch.bmm(weights, context)
        output = torch.bmm(weights.transpose(1, 2), context)
        output = output.squeeze(dim = 1)
        return output, weights



class TitleEntityEncoder(nn.Module):
    def __init__(self, config, word_emb_size, hidden_size):
        super().__init__()



    def forward(self, encoded_text, title_map):
        title_vectors = map2vec(encoded_text, title_map)
        encoded_title = title_vectors.sum(dim = 1)/title_map.sum((1,2)).view(-1,1)

        return encoded_title

class MentionEncoder(nn.Module):
    def __init__(self,word_embedding_size,hidden_size):
        super().__init__()


    def forward(self, encoded_text, em_mapping):
        text_shape = encoded_text.shape
        emm_shape = em_mapping.shape

        output_shape = emm_shape[:-1] + text_shape[-1:]
        em_vectors = torch.bmm(em_mapping.view(emm_shape[0], -1, emm_shape[-1]), encoded_text).view(output_shape)

        return em_vectors


class TREX_Base(nn.Module):
    PREDICTION_OPT_PREDICT_MAX_SINGLE = 1
    PREDICTION_OPT_ATTN_PREDICT = 2
    PREDICTION_OPT_PREDICT_MAX = 3

    PREDICTION_OPT_PREDICT_STOCHASTIC = 4
    PREDICTION_OPT_PREDICT_SMOOTHMAX = 5

    PREDICTION_OPT_AVERAGE_PREDICT = 6

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_dropout = True
        self.use_bert = config.use_bert
        self.outputsize = config.relation_num
        self.prediction_opt = config.prediction_opt
        self.softmax1 = torch.nn.Softmax(dim = 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.drop_prob = 0.5
        self.drop_prob_trf = 0.2

        

        hidden_size = config.hidden_size
        self.text_encoder_name = "bert"
        word_embedding_size = self.init_text_encoding_layer(hidden_size)
        hidden_size = word_embedding_size
        #entity_emb_size = self.init_title_entity_encoder(word_embedding_size)
        #entity_emb_size = self.init_entity_mention_encoder(word_embedding_size)
        self.title_entity_encoder = TitleEntityEncoder(config, word_embedding_size,hidden_size)
        self.entity_mention_encoder = MentionEncoder(word_embedding_size,hidden_size)

        self.init_prediction_module(hidden_size, config.relation_num)

        if self.use_bert:
            self.dropout = nn.Dropout(self.drop_prob_trf)
        else:
            self.dropout = nn.Dropout(self.drop_prob)


    def init_prediction_module(self, hidden_size, relation_num):
        self.prediction_bilinear = nn.Bilinear(hidden_size, hidden_size, relation_num)

        if self.prediction_opt == self.PREDICTION_OPT_ATTN_PREDICT:
            self.prediction_attention = AttentionModelLargeMemory(hidden_size, hidden_size,hidden_size)


    @conditional_profiler
    def forward(self, context_idxs, context_ner, context_char_idxs, title_map, em_map, label_mask, label_mask_m, mention_cumsum, context_starts = None, context_masks = None, deterministic=False):
        if self.use_bert:
            context_output = self.encode_text(context_idxs, context_ner, context_masks, context_starts)
        else:
            context_output = self.encode_text(context_idxs, context_ner)

        title_vector = self.title_entity_encoder(context_output, title_map)
        em_vectors = self.entity_mention_encoder(context_output, em_map)

        if self.use_dropout:
            title_vector = self.dropout(title_vector)
            em_vectors = self.dropout(em_vectors)

        batch_size, hidden_size = title_vector.shape
        # is_logit = True
        # if self.prediction_opt == self.PREDICTION_OPT_PREDICT_SMOOTHMAX:
        #     is_logit = False

        if self.prediction_opt == self.PREDICTION_OPT_PREDICT_MAX_SINGLE:
            batch_size, ent_cnt, mention_cnt, _ = em_map.shape
            title_vector_expand = title_vector.view(batch_size, 1,1, hidden_size).expand((-1, ent_cnt, mention_cnt, -1)).contiguous()
            shift4masking = (10000 * (1 - label_mask)).unsqueeze(-1)

            tmp_h_full = self.prediction_bilinear(em_vectors, title_vector_expand)
            tmp_t_full = self.prediction_bilinear(title_vector_expand, em_vectors)
            tmp_h_full = tmp_h_full - shift4masking
            tmp_t_full = tmp_t_full - shift4masking
            rels_h = torch.max(tmp_h_full, dim=2)[0]
            rels_t = torch.max(tmp_t_full, dim=2)[0]

        elif self.prediction_opt == self.PREDICTION_OPT_ATTN_PREDICT:
            num_entities = len(mention_cumsum)
            title_vector_expand = title_vector.view(batch_size, 1, hidden_size).expand(
                (-1, num_entities, -1)).contiguous()

            ent_vectors = []
            range_begin = 0
            for eid, range_end in enumerate(mention_cumsum):
                ent_vector, weight = self.prediction_attention(title_vector, em_vectors[:, range_begin:range_end],label_mask_m[:,range_begin:range_end])

                ent_vectors.append(ent_vector)
                range_begin = range_end

            ent_vectors = torch.stack(ent_vectors, dim=1)
            rels_h = self.prediction_bilinear(title_vector_expand, ent_vectors)
            rels_t = self.prediction_bilinear(ent_vectors, title_vector_expand)
        elif self.prediction_opt == self.PREDICTION_OPT_AVERAGE_PREDICT:
            num_entities = len(mention_cumsum)
            title_vector_expand = title_vector.view(batch_size, 1, hidden_size).expand(
                (-1, num_entities, -1)).contiguous()

            ent_vectors = []
            range_begin = 0
            for eid, range_end in enumerate(mention_cumsum):
                #ent_vector, weight = self.prediction_attention(title_vector, em_vectors[:, range_begin:range_end],label_mask_m[:,range_begin:range_end])
                ent_vector = torch.mean(em_vectors[:, range_begin:range_end],dim = 1)
                ent_vectors.append(ent_vector)
                range_begin = range_end
            ent_vectors = torch.stack(ent_vectors, dim=1)
            rels_h = self.prediction_bilinear(title_vector_expand, ent_vectors)
            rels_t = self.prediction_bilinear(ent_vectors, title_vector_expand)

        else:
            msize = mention_cumsum[-1].item()
            title_vector_expand = title_vector.view(batch_size, 1, hidden_size).expand(
                (-1, msize, -1)).contiguous()

            shift4masking = (10000 * (1 - label_mask_m))

            tmp_h_full = self.prediction_bilinear(em_vectors, title_vector_expand)
            tmp_t_full = self.prediction_bilinear(title_vector_expand, em_vectors)
            tmp_h_full = tmp_h_full - shift4masking
            tmp_t_full = tmp_t_full - shift4masking

            res_h_list = []
            res_t_list = []

            range_begin = 0
            for eid, range_end in enumerate(mention_cumsum):
                res_h = tmp_h_full[:, range_begin:range_end]
                res_t = tmp_t_full[:, range_begin:range_end]

                if self.prediction_opt == self.PREDICTION_OPT_PREDICT_MAX:
                    res_h = torch.max(res_h, dim = 1)[0]
                    res_t = torch.max(res_t, dim = 1)[0]
                elif self.prediction_opt == self.PREDICTION_OPT_PREDICT_SMOOTHMAX:
                    res_h = torch.logsumexp(res_h, dim = 1)
                    res_t = torch.logsumexp(res_t, dim = 1)

                elif self.prediction_opt == self.PREDICTION_OPT_PREDICT_STOCHASTIC:
                    weight_h = self.softmax1(res_h)
                    weight_h = weight_h * label_mask[:, eid].unsqueeze(-1)
                    res_h = res_h * weight_h
                    res_h = torch.sum(res_h, dim=1)

                    weight_t = self.softmax1(res_t)
                    weight_t = weight_t * label_mask[:, eid].unsqueeze(-1)
                    res_t = res_t * weight_t
                    res_t = torch.sum(res_t, dim=1)

                res_h_list.append(res_h)
                res_t_list.append(res_t)
                range_begin = range_end

            rels_h = torch.stack(res_h_list, dim=1)
            rels_t = torch.stack(res_t_list, dim=1)

        return rels_h, rels_t


class TREX_Simple(TREX_Base):
    name = "TREX_Simple"

    def init_basernn(self, config, hidden_size, n_layers):
        input_size = config.data_word_vec.shape[1] + config.entity_type_size
        self.rnn_base = nn.LSTM(input_size, hidden_size, n_layers, bias=True, batch_first=True, bidirectional=True)


    def init_text_encoding_layer(self, hidden_size):
        config = self.config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        self.word_emb.weight.requires_grad = False
        self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.text_encoder_name== "BiLSTM":
            self.init_basernn(config, hidden_size, 1)
            return hidden_size *2

        return hidden_size


    @conditional_profiler
    def encode_text(self, context_idxs, context_ner):
        word_embedding = self.word_emb(context_idxs)
        if self.use_dropout:
            word_embedding = self.dropout(word_embedding)
        sent = torch.cat([word_embedding, self.ner_emb(context_ner)], dim=-1)
        if self.text_encoder_name== "BiLSTM":
            context_output, _ = self.rnn_base(sent)
            maxlen, batchsize, hiddensize = context_output.size()

        return context_output


class TREX(TREX_Base):
    name = "T-REX"
    def __init__(self, config):
        super().__init__(config)

        if self.use_dropout:
            self.dropout_trf = nn.Dropout(self.drop_prob_trf)


    def init_text_encoding_layer(self, hidden_size):
        config = self.config

        if self.text_encoder_name == "bert":
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            transformer_hidden_size = 768
        else:
            self.transfoxl = TransfoXLModel.from_pretrained('transfo-xl-wt103')

        self.linear = nn.Linear(transformer_hidden_size, hidden_size)
        return hidden_size

    def encode_text(self, context_idxs, context_ner, context_masks, context_starts):
        bert_out = self.bert(context_idxs, attention_mask=context_masks)[0]
        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(bert_out, context_starts)]
        # print('output_2',context_output[0])
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)

        # print('output_3',context_output[0])
        # print(context_output.size())
        context_output = torch.nn.functional.pad(context_output,
                                                 (0, 0, 0, context_idxs.size(-1) - context_output.size(-2)))

        if self.use_dropout:
            context_output = self.dropout_trf(context_output)
        # context_output = bert_out
        context_output = self.linear(context_output)

        return context_output
