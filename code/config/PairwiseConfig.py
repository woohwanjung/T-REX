import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import time
import datetime
import json
import sklearn.metrics

#from tensorboardX import SummaryWriter

from config.BaseConfig import BaseConfig, Accuracy
from config.Config import IGNORE_INDEX
from config.TREXConfig import TopicAwareConfig, RESULT_OPT_PUBLISH
from settings import *
from utils import dict_set


class PairwiseConfig(TopicAwareConfig):
    name = "Pairwise"
    def __init__(self, args):
        super().__init__(args)



    def forward(self, model, data):
        context_idxs = data['context_idxs']
        context_pos = data['context_pos']
        h_mapping = data['h_mapping']
        t_mapping = data['t_mapping']
        #relation_label = data['relation_label']
        input_lengths = data['input_lengths']
        #relation_multi_label = data['relation_multi_label']
        relation_mask = data['relation_mask']
        context_ner = data['context_ner']
        #context_char_idxs = data['context_char_idxs']
        ht_pair_pos = data['ht_pair_pos']
        sent_idxs = data['sent_idxs']
        sent_lengths = data['sent_lengths']
        reverse_sent_idxs = data['reverse_sent_idxs']

        if "context_char_idxs" in data:
            context_char_idxs = data["context_char_idxs"]
        else:
            context_char_idxs = None


        dis_h_2_t = ht_pair_pos + 10
        dis_t_2_h = -ht_pair_pos + 10

        if self.use_bert:
            context_masks = data['context_masks']
            context_starts = data['context_starts']
            predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,
                               h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs,
                               sent_lengths, reverse_sent_idxs, context_masks, context_starts)
        else:
            predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)
        return predict_re



    def get_loss(self, output, data):
        #predict_re, relation_label, relation_multi_label, relation_mask, BCE)

        relation_mask = data['relation_mask'].unsqueeze(2)
        relation_multi_label = data['relation_multi_label']

        loss_mat = self.bce_loss_logits(output, relation_multi_label)* relation_mask
        loss = torch.sum(loss_mat)
        loss = loss / (self.relation_num * torch.sum(relation_mask))

        return loss

    def get_result(self, model, data_opt="test", theta=0.5, result_opt=RESULT_OPT_PUBLISH):
        if data_opt == "train":
            dat_file = self.train_file
            get_batches = self.get_train_batch_np
            dat_len = self.train_len
            title_entities = self.train_title_entities
            #dat_emid2eidmid = self.train_emid2eidmid
        elif data_opt == "test":
            dat_file = self.test_file
            get_batches = self.get_test_batch
            dat_len = self.test_len
            title_entities = self.test_title_entities
            #dat_emid2eidmid = self.test_emid2eidmid


        sigmoid = nn.Sigmoid()

        if result_opt == RESULT_OPT_PUBLISH:
            result = [dict_set() for _ in range(dat_len)]
        #else:
        #    result_by_doc = [dict_max() for _ in range(dat_len)]

        model.eval()
        for data in get_batches():
            doc_ids = data['doc_ids']
            #n_ems = data['n_ems']

            output_rels_l = self.forward(model, data)
            batch_size = len(doc_ids)


            # OPT_PUB
            output_rels = sigmoid(output_rels_l)
            res = (output_rels >= theta).cpu().numpy()

            for i, doc_id in enumerate(doc_ids):
                title_vertex_id, title_vertex = title_entities[doc_id][0]

                doc = dat_file[doc_id]
                vertexSet = doc['vertexSet']
                result_d = result[doc_id]

                L = len(vertexSet)
                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx == t_idx:
                            continue
                        if h_idx == title_vertex_id:
                            rh_list = res[i, j].nonzero()[0].tolist()
                            for r_h in rh_list:
                                if r_h > 0:
                                    rec = (title_vertex_id, t_idx, r_h)
                                    result_d.add(rec, -1)

                        elif t_idx == title_vertex_id:
                            rt_list = res[i, j].nonzero()[0].tolist()
                            for r_t in rt_list:
                                if r_t > 0:
                                    rec = (h_idx, title_vertex_id, r_t)
                                    result_d.add(rec, -1)
                        j+=1



        return result


    def get_test_batch(self, contain_relation_multi_label = False):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()
        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()
        sent_idxs = torch.LongTensor(self.test_batch_size, self.sent_limit, self.word_size).cuda()
        reverse_sent_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()

        context_masks = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.test_batch_size, self.max_length).cuda()

        relation_multi_label = torch.FloatTensor(self.test_batch_size, self.h_t_limit, self.relation_num).cuda()

        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()


            ht_pair_pos.zero_()

            sent_idxs.zero_()
            sent_idxs -= 1
            reverse_sent_idxs.zero_()
            reverse_sent_idxs -= 1

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x]>0) , reverse = True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []
            doc_ids = []
            evi_nums = []

            for i, index in enumerate(cur_batch):
                doc_ids.append(index)
                if self.use_bert:
                    context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_word[index, :]))
                    context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))
                    context_starts[i].copy_(torch.from_numpy(self.data_test_bert_starts[index, :]))
                else:
                    context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))

                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]
                this_sent_idxs, this_reverse_sent_idxs = self.load_sent_idx(ins)
                sent_idxs[i].copy_(torch.from_numpy(this_sent_idxs))
                reverse_sent_idxs[i].copy_(torch.from_numpy(this_reverse_sent_idxs))

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]


                            for h in hlist:
                                h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                            for t in tlist:
                                t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                            relation_mask[i, j] = 1

                            delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                            #this is for qualitative analysis
                            if contain_relation_multi_label:
                                label = idx2label[(h_idx, t_idx)]
                                for r in label:
                                    relation_multi_label[i, j, r] = 1

                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                evi_num_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in'+self.train_prefix]
                    evi_num_set[(label['h'], label['t'], label['r'])] = len(label['evidence'])

                labels.append(label_set)
                evi_nums.append(evi_num_set)


                L_vertex.append(L)
                indexes.append(index)



            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            sent_lengths = (sent_idxs[:cur_bsz] > 0).long().sum(-1)


            batch =  {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths,
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes,
                   'sent_idxs': sent_idxs[:cur_bsz],
                   'sent_lengths': sent_lengths[:cur_bsz],
                   'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
                   'evi_num_set': evi_nums,
                    "doc_ids":doc_ids,
                   }

            if self.use_bert:
                batch['context_masks']= context_masks[:cur_bsz, :max_c_len].contiguous()
                batch['context_starts']= context_starts[:cur_bsz, :max_c_len].contiguous()
            if contain_relation_multi_label:
                batch["relation_multi_label"] = relation_multi_label[:cur_bsz, :max_h_t_cnt]
            yield batch


    def _get_train_batch(self,b, train_order):
        data_len = self.train_len
        start_id = b * self.batch_size

        cur_bsz = min(self.batch_size, data_len - start_id)

        cur_batch = [(i, np.sum(self.data_train_word[i]>0).item()) for i in train_order[start_id: start_id + cur_bsz]]
        cur_batch.sort(key = lambda x: x[1], reverse=True)
        max_length = cur_batch[0][1]

        if self.use_bert:
            sent_lengths = []
            for i, len_w in cur_batch:
                if self.use_bert:
                    sent_lengths.append(np.where(self.data_train_bert_word[i]==102)[0].item()+1)
                else:
                    sent_lengths.append(len_w)

            max_length = max(sent_lengths)

        sent_limit = self.sent_limit

        shape_txt = (cur_bsz, max_length)
        shape_pair = (cur_bsz, self.h_t_limit)

        context_idxs = np.zeros(shape_txt, dtype=np.int)
        context_ner = np.zeros(shape_txt, dtype=np.int)
        context_pos = np.zeros(shape_txt, dtype=np.int)
        pos_idx = np.zeros(shape_txt, dtype=np.int)

        if self.use_bert:
            context_masks = np.zeros(shape_txt, dtype=np.int)
            context_starts = np.zeros(shape_txt, dtype=np.int)

        # context_char_idxs = np.zeros(shape_txt + (self.char_limit,), dtype=np.int)
        if HALF_PRECISION:
            float_type = np.float32
        else:
            float_type = np.float

        shape_b_ht_l = shape_pair + (max_length,)
        h_mapping = np.zeros(shape_b_ht_l, dtype = float_type)
        t_mapping = np.zeros(shape_b_ht_l, dtype = float_type)

        relation_mask = np.zeros(shape_pair, dtype = float_type)
        relation_label = np.full(shape_pair, IGNORE_INDEX, dtype=np.int)
        relation_multi_label = np.zeros(shape_pair + (self.relation_num,), dtype = float_type)
        ht_pair_pos = np.zeros(shape_pair, dtype=np.int)

        sent_idxs = np.full((cur_bsz, sent_limit, self.word_size), -1, dtype=np.int)
        reverse_sent_idxs = np.full((cur_bsz, max_length), -1, dtype=np.int)

        common_shape_bc = (2 * cur_bsz, self.train_max_entities, self.relation_num)
        ht_label = np.zeros(common_shape_bc)

        max_h_t_cnt = 1
        L_vertex = []
        indexes = []
        vertex_sets = []
        for i, (index, _) in enumerate(cur_batch):
            data_file = self.train_file
            data_word = self.data_train_word
            data_pos = self.data_train_pos
            data_ner = self.data_train_ner
            if self.use_bert:
                data_bert_word = self.data_train_bert_word
                data_bert_mask = self.data_train_bert_mask
                data_bert_starts = self.data_train_bert_starts

            for j in range(max_length):
                if data_word[index, j] == 0:
                    break
                pos_idx[i, j] = j + 1

            if self.use_bert:
                doclen = sent_lengths[i]
                context_idxs[i, :doclen] = data_bert_word[index, :doclen]
                context_masks[i, :doclen] = data_bert_mask[index, :doclen]
                context_starts[i, :doclen] = data_bert_starts[index, :doclen]
                #cs_idx = context_starts[i, :doclen].nonzero()[0]
                #context_starts_idx[i,:len(cs_idx)] = cs_idx

            else:
                doclen = (data_word[index] > 0).sum()
                context_idxs[i, :doclen] = data_word[index, :doclen]

            context_pos[i, :doclen] = data_pos[index, :doclen]
            context_ner[i, :doclen] = data_ner[index, :doclen]
            # context_char_idxs[i, :doclen] = data_char[index, :doclen]

            ins = data_file[index]
            this_sent_idxs, this_reverse_sent_idxs = self.load_sent_idx(ins)
            sent_idxs[i, :sent_limit] = this_sent_idxs[:sent_limit]
            reverse_sent_idxs[i, :max_length] = this_reverse_sent_idxs[:max_length]

            labels = ins['labels']
            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])

            train_tripe = list(idx2label.keys())
            for j, (h_idx, t_idx) in enumerate(train_tripe):
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                for h in hlist:
                    h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                for t in tlist:
                    t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                label = idx2label[(h_idx, t_idx)]

                delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                for r in label:
                    relation_multi_label[i, j, r] = 1

                relation_mask[i, j] = 1
                rt = np.random.randint(len(label))
                relation_label[i, j] = label[rt]


            # random.shuffle(ins['na_triple'])
            # lower_bound = max(20, len(train_tripe)*3)
            if self.use_bert:
                lower_bound = max(20, len(train_tripe) * 3)
                lower_bound = min(len(ins['na_triple']), lower_bound)
            else:
                lower_bound = len(ins['na_triple'])
            sel_idx = random.sample(list(range(len(ins['na_triple']))), min(len(ins['na_triple']), lower_bound))
            sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]
            # sel_ins = []
            # for j, (h_idx, t_idx) in enumerate(ins['na_triple'], len(train_tripe)):
            for j, (h_idx, t_idx) in enumerate(sel_ins, len(train_tripe)):
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                for h in hlist:
                    h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                for t in tlist:
                    t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                relation_multi_label[i, j, 0] = 1
                relation_label[i, j] = 0
                relation_mask[i, j] = 1
                delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
            # print(max_h_t_cnt)

            max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

        input_lengths = (context_idxs[:cur_bsz] > 0).sum(1)
        max_c_len = int(input_lengths.max())

        # print(".")#assert max_c_len == max_length
        max_c_len = max(max_c_len, max_length)

        sent_lengths = []
        for i in cur_batch:
            if self.use_bert:
                len_w = np.where(self.data_train_bert_word[i] == 102)[0]
                if len_w.shape[0] == 0:
                    len_w = max_length
                else:
                    len_w = len_w.item() + 1

            else:
                len_w = np.sum(self.data_train_word[i] > 0).item()
            sent_lengths.append(len_w)
        max_length = max(sent_lengths)
        sent_lengths = np.array(sent_lengths, dtype = np.int)

        batch = {
            "torch": False,
            'context_idxs': context_idxs[:cur_bsz, :max_c_len],
            'context_pos': context_pos[:cur_bsz, :max_c_len],
            'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
            't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
            'relation_label': relation_label[:cur_bsz, :max_h_t_cnt],
            'input_lengths': input_lengths,
            'pos_idx': pos_idx[:cur_bsz, :max_c_len],
            'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
            'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
            'context_ner': context_ner[:cur_bsz, :max_c_len],
            # 'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len],
            'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
            'sent_idxs': sent_idxs[:cur_bsz],
            'sent_lengths': sent_lengths[:cur_bsz],
            'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
            'cur_batch':cur_batch,
            'cur_bsz':cur_bsz,
        }
        if self.use_bert:
            batch['context_masks'] = context_masks[:cur_bsz, :max_c_len]
            batch['context_starts'] = context_starts[:cur_bsz, :max_c_len]
            #batch['context_starts_idx'] = context_starts_idx[:cur_bsz, :max_c_len]


        return batch

    def full_test(self, model, input_theta = -1.0):
        result_title_ext = []

        dat_file = self.test_file
        get_batches = self.get_test_batch
        dat_len = self.test_len
        title_entities = self.test_title_entities

        sigmoid = nn.Sigmoid()

        model.eval()
        for data in get_batches():
            doc_ids = data['doc_ids']
            output_rels_l = self.forward(model, data)

            batch_size = len(doc_ids)

            # OPT_PUB
            output_rels = sigmoid(output_rels_l)
            res = output_rels.detach().cpu().numpy()

            for i in range(batch_size):
                idx = doc_ids[i]
                doc = dat_file[idx]
                title_vertex_id, title_vertex = title_entities[idx][0]
                vertexSet = doc["vertexSet"]

                L = len(vertexSet)
                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx == t_idx:
                            continue
                        if h_idx == title_vertex_id or t_idx == title_vertex_id:
                            for r in range(1, self.relation_num):
                                result_title_ext.append((res[i, j, r].item(), idx, h_idx, r, t_idx))

                        j += 1

        result_title_ext.sort(key = lambda v:v[0], reverse= True)

        num_labels = 0
        num_labels_ign = 0
        label_set = {}

        key_intrain = "in"+ self.train_prefix

        for idx, doc in enumerate(dat_file):
            title_vertex_id, title_vertex = title_entities[idx][0]
            for label in doc['labels']:
                h = label['h']
                t = label['t']
                r = label['r']
                if title_vertex_id == h or title_vertex_id == t:
                    num_labels += 1
                    label_set[(idx, h, r, t)] = label[key_intrain]
                    if len(label_set) != num_labels:
                        print("Dup")

                    if not label[key_intrain]:
                        num_labels_ign += 1


        assert len(label_set) == num_labels
        #print(len(result_title_ext))
        precision_list = []
        precision_ign_list = []
        recall_list = []
        recall_ign_list = []
        n_tp = 0
        n_tp_ign = 0

        f1_pos_th = 0

        i_ign = 0
        for i in range(len(result_title_ext)):
            conf, doc_id, h, r, t = result_title_ext[i]
            key = (doc_id, h, r, t)
            correct =  key in label_set

            if correct:
                n_tp += 1
                if not label_set[key]:
                    n_tp_ign += 1

            precision = n_tp/(i+1)
            recall = n_tp/num_labels
            precision_list.append(precision)
            recall_list.append(recall)
            if conf > input_theta:
                f1_pos_th = i

            if correct and label_set[key]:
                continue

            precision_ign = n_tp_ign / (i_ign+1)
            recall_ign = n_tp_ign / num_labels_ign

            precision_ign_list.append(precision_ign)
            recall_ign_list.append(recall_ign)
            if conf > input_theta:
                f1_pos_th = i_ign
            i_ign += 1

        pr_x = np.array(recall_list)
        pr_y = np.array(precision_list)
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        if input_theta < 0:
            f1_pos = f1_arr.argmax()
        else:
            f1_pos = f1_pos_th

        precision = precision_list[f1_pos]
        recall = recall_list[f1_pos]
        f1 = f1_arr[f1_pos]
        opt_theta = result_title_ext[f1_pos][0]

        pr_x_ign = np.array(recall_ign_list)
        pr_y_ign = np.array(precision_ign_list)
        auc_ign = sklearn.metrics.auc(x=pr_x_ign, y=pr_y_ign)

        f1_ign_arr = (2 * pr_x_ign * pr_y_ign / (pr_x_ign + pr_y_ign + 1e-20))


        th_ign = input_theta if input_theta >0.0 else opt_theta
        i_ign = 0
        for i in range(len(result_title_ext)):
            conf, doc_id, h, r, t = result_title_ext[i]
            key = (doc_id, h, r, t)
            correct = key in label_set

            if correct and label_set[key]:
                continue
            if conf < th_ign:
                break
            i_ign += 1


        f1_ign = f1_ign_arr[i_ign]


        res_dict = {
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc,
            "opt_theta": opt_theta,
            "F1Ign": f1_ign,
            "AUCIgn": auc_ign,
        }

        if input_theta < 0.0:
            print("Optimal theta", opt_theta)

        return res_dict

    def find_theta(self, model):
        result_title_ext = []

        dat_file = self.test_file
        get_batches = self.get_test_batch
        dat_len = self.test_len
        title_entities = self.test_title_entities


        sigmoid = nn.Sigmoid()


        model.eval()
        for data in get_batches():
            doc_ids = data['doc_ids']
            output_rels_l = self.forward(model, data)

            batch_size = len(doc_ids)


            # OPT_PUB
            output_rels = sigmoid(output_rels_l)
            res = output_rels.detach().cpu().numpy()


            for i in range(batch_size):
                idx = doc_ids[i]
                doc = dat_file[idx]
                title_vertex_id, title_vertex = title_entities[idx][0]
                vertexSet = doc["vertexSet"]

                L = len(vertexSet)
                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx == t_idx:
                            continue
                        if h_idx == title_vertex_id or t_idx == title_vertex_id:
                            for r in range(1, self.relation_num):
                                result_title_ext.append((res[i, j, r].item(), idx, h_idx, r, t_idx))

                        j += 1



        result_title_ext.sort(key = lambda v:v[0], reverse= True)

        num_labels = 0
        label_set = set()
        for idx, doc in enumerate(dat_file):
            title_vertex_id, title_vertex = title_entities[idx][0]
            for label in doc['labels']:
                h = label['h']
                t = label['t']
                r = label['r']
                if title_vertex_id == h or title_vertex_id == t:
                    num_labels += 1
                    label_set.add((idx, h, r, t))
                    if len(label_set) != num_labels:
                        print("Dup")

        assert len(label_set) == num_labels
        print(len(result_title_ext))
        precision_list = []
        recall_list = []
        n_tp = 0

        for i in range(len(result_title_ext)):
            conf, doc_id, h, r, t = result_title_ext[i]
            if (doc_id, h, r, t) in label_set:
                n_tp += 1

            precision = n_tp/(i+1)
            recall = n_tp/num_labels
            precision_list.append(precision)
            recall_list.append(recall)

        pr_x = np.array(recall_list)
        pr_y = np.array(precision_list)
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        #auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        theta = result_title_ext[f1_pos][0]
        print(f"F1:{f1}, Theta: {theta}")
        return f1, theta





