import datetime
import json
import pickle
import random

import numpy as np
import sklearn
import torch
TENSORBOAD_LOADED = True
try:
    from tensorboardX import SummaryWriter
except:
    TENSORBOAD_LOADED = False
from torch import nn


from config.BaseConfig import BaseConfig, Accuracy, DummyWriter
from models.TREX import TREX_Base
from settings import *
# from config.ExtConfig import WRITER_NAME
from utils import dict_set

IGNORE_LABEL = -100
EM_INCLUDE_TITLE = False

RESULT_OPT_PUBLISH = 0
RESULT_OPT_ALL_PROB = 1



def opt2testprefix(split_dev, dev):
    if split_dev:
        if dev:
            test_prefix = "dev_dev1"
        else:
            test_prefix = "dev_dev2"
    else:
        if dev:
            test_prefix = "dev_dev"
        else:
            test_prefix = "dev_test"
    return test_prefix

def testprefix2opt(test_prefix):
    if test_prefix == "dev_test":
        split_dev = False
        dev = False
    elif test_prefix == "dev_dev":
        split_dev = False
        dev = True
    else:
        split_dev = True
        if test_prefix == "dev_dev1":
            dev = True
        else:
            dev = False
    return split_dev, dev




def get_expname(option_dict):
    model_name = option_dict['model_name']
    learning_rate = option_dict['learning_rate']
    train_prefix = option_dict['train_prefix']
    return f"{train_prefix}_{model_name}_lr{learning_rate}_h{option_dict['hidden_size']}"


def find_title_entities(datafile):
    def canonical_form(title_str):
        name = title_str.lower()
        name = name.replace(" - ", "-").replace(" '", "'")

        return name

    num_docs = len(datafile)
    has_title_entity = 0
    n_title_entity = 0

    n_matching = 0
    n_canonical_matching = 0
    n_canonical_start = 0
    n_canonical_overlap = 0

    title_list = []
    title_entities = [[] for _ in range(num_docs)]

    for i, doc in enumerate(datafile):
        title = doc['title']
        title_canonical = canonical_form(title)
        title_list.append(title)
        vertexSet = doc['vertexSet']
        matched = False
        for vi, entity in enumerate(vertexSet):
            if matched:
                continue
            for em in entity:
                if canonical_form(em['name']) == title_canonical:
                    matched = True
                    title_entities[i].append((vi, entity))
                    n_title_entity += 1
                    n_canonical_matching += 1
                    if vi > 0:
                        pass
                    # print(title, em)
                    break

        if matched:
            has_title_entity += 1
            continue

        for vi, entity in enumerate(vertexSet):
            if matched:
                continue
            for em in entity:
                if title_canonical.startswith(canonical_form(em['name'])):
                    title_entities[i].append((vi, entity))
                    matched = True
                    n_canonical_start += 1
                    n_title_entity += 1
                    break
                    if vi > 0:
                        pass
                    # print(title)
                    # print(em["name"])

        if matched:
            has_title_entity += 1
            continue

        title_bow = set(title_canonical.split())
        for vi, entity in enumerate(vertexSet):
            if matched:
                continue
            for em in entity:
                em_canonical = canonical_form(em['name'])
                em_bow = set(em_canonical.split())

                overlap = title_bow & em_bow
                if len(overlap) > 0:
                    title_entities[i].append((vi, entity))
                    matched = True
                    n_canonical_overlap += 1
                    n_title_entity += 1
                    break

        if matched:
            has_title_entity += 1
            continue

        # print(i,title)
        # print(vertexSet[0][0]['name'])
        # print("-----")

        title_entities[i].append((0, vertexSet[0]))

    print(num_docs, has_title_entity, n_title_entity)
    print("Matching", n_canonical_matching)
    print("Start", n_canonical_start)
    print("Overlap", n_canonical_overlap)
    return title_entities


def get_datafile_single(datafile, title_entities):
    data_out = []

    for did in range(len(datafile)):
        h_labels = []
        t_labels = []
        dat = {"h_labels": h_labels, "t_labels": t_labels}
        if len(title_entities[did]) == 0:
            data_out.append(dat)
            print("Warning no title entities", did)
            continue
        if len(title_entities[did]) > 1:
            print("Warning multiple title entities", did)

        vid_title_entity = title_entities[did][0][0]
        ins = datafile[did]

        for label in ins['labels']:
            if label['h'] == vid_title_entity:
                h_labels.append(label)

            elif label['t'] == vid_title_entity:
                t_labels.append(label)

        data_out.append(dat)
    return data_out



class TopicAwareConfig(BaseConfig):
    name = "TopicREConfig"

    def __init__(self, args):
        super().__init__(args)
        self.hidden_size = args.hidden_size
        self.prediction_opt = args.prediction_opt

        self.batch_keys_float+=["label_mask","label_mask_m","title_map","em_map","ht_label"]

        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")

        self.best_f1 = 0.0
        self.best_epoch = 0


    def set_entity_mentions(self, dat_file, title_entities):
        max_title_mentions = 0
        max_entities = 0
        max_mentions_per_entity = 0
        num_entity_mentions = []

        e2labels_h_all = []
        e2labels_t_all = []

        emid2eidmid_all = []
        eidmid2emid_all = []
        em2labels = []
        for did, doc in enumerate(dat_file):
            eidmid2emid = {}
            emid2eidmid = []
            em2labels_h = {}
            em2labels_t = {}
            title_id, title_entity = title_entities[did][0]
            max_title_mentions = max(max_title_mentions, len(title_entity))

            vertexSet = doc['vertexSet']
            max_entities = max(max_entities, len(vertexSet))


            e2labels_h = [[] for _ in vertexSet]
            e2labels_t = [[] for _ in vertexSet]
            em_id = 0
            for eid, entity in enumerate(vertexSet):
                if eid == title_id and not EM_INCLUDE_TITLE:
                    continue
                max_mentions_per_entity = max(max_mentions_per_entity, len(entity))
                for mid in range(len(entity)):
                    emid2eidmid.append((eid, mid))
                    eidmid2emid[(eid, mid)] = em_id
                    em_id += 1

            emid2eidmid_all.append(emid2eidmid)
            num_entity_mentions.append(em_id)

            for label in doc['labels']:
                h = label['h']
                t = label['t']
                r = label['r']

                if h == title_id:
                    e2labels_h[t].append(r)
                elif t == title_id:
                    e2labels_t[h].append(r)
                else:
                    continue
            e2labels_h_all.append(e2labels_h)
            e2labels_t_all.append(e2labels_t)

        return e2labels_h_all, e2labels_t_all, max_title_mentions, max_entities, max_mentions_per_entity



    def load_train_data(self):
        super().load_train_data()
        title_entities = find_title_entities(self.train_file)
        self.train_title_entities = title_entities
        self.train_file_single = get_datafile_single(self.train_file, self.train_title_entities)

        e2labels_h, e2labels_t, max_title_mentions, max_entities, max_mentions_per_entity \
            = self.set_entity_mentions(self.train_file, title_entities)


        self.train_e2labels_h = e2labels_h
        self.train_e2labels_t = e2labels_t
        self.train_max_title_mentions = max_title_mentions
        self.train_max_entities = max_entities
        self.train_max_mentions_per_entity = max_mentions_per_entity

        self.train_did_len = []
        for i in range(len(self.train_file)):
            self.train_did_len.append((i, len(self.train_file[i])))



    def load_test_data(self, split_dev = True, dev = True):
        self.test_prefix = opt2testprefix(split_dev, dev)
        super().load_test_data()
        title_entities = find_title_entities(self.test_file)
        self.test_title_entities = title_entities
        self.test_file_single = get_datafile_single(self.test_file, self.test_title_entities)
        e2labels_h, e2labels_t, max_title_mentions, max_entities, max_mentions_per_entity \
            = self.set_entity_mentions(self.test_file, title_entities)

        self.test_e2labels_h = e2labels_h
        self.test_e2labels_t = e2labels_t
        self.test_max_title_mentions = max_title_mentions
        self.test_max_entities = max_entities
        self.test_max_mentions_per_entity = max_mentions_per_entity


    def publish(self, model, fname, theta = -1):
        if theta <0.0:
            theta = self.find_theta(model)

        result = self.get_result(model,theta = theta)

        output_data = []
        output_data_wikidataid = []

        for did, result_d in enumerate(result):
            title = self.test_file[did]['title']

            for h, t, r in result_d:
                rec = {"title": title, "h_idx": h, "t_idx": t, "r": r, "evidence": []}
                output_data.append(rec)

                rec_w = rec.copy()
                rec_w["r"] = self.id2rel[r]
                output_data_wikidataid.append(rec_w)

        if not os.path.exists(f"{EXTRACTION_DIR}"):
            os.makedirs(f"{EXTRACTION_DIR}")
        with open(f"{EXTRACTION_DIR}/{self.test_prefix}_{fname}.json", "w") as f:
             json.dump(output_data,f)

        with open(f"{EXTRACTION_DIR}/{self.test_prefix}_{fname}_w.json", "w") as f:
             json.dump(output_data_wikidataid,f)

    def publish_with_confidence(self, model, fname, data_opt):
        result = self.get_result(model, theta = -0.1, data_opt = data_opt)

        if data_opt == "train":
            dat_file = self.train_file
            get_batches = self.get_train_batch_np
            dat_len = self.train_len
            title_entities = self.train_title_entities
            # dat_emid2eidmid = self.train_emid2eidmid
        elif data_opt == "test":
            dat_file = self.test_file
            get_batches = self.get_test_batch_np
            dat_len = self.test_len
            title_entities = self.test_title_entities
            # dat_emid2eidmid = self.test_emid2eidmid
        # dat_emid2eidmid = self.test_emid2eidmid

        softmax = nn.Softmax(dim=-1)
        sigmoid = nn.Sigmoid()

        #dht_list = []
        #conf_list = []
        result_dict = {}

        model.eval()
        for data in get_batches():
            doc_ids = data['doc_ids']
            # n_ems = data['n_ems']

            output_rels_h, output_rels_t = self.forward(model, data)
            # output_rels = torch.cat((output_rels_h, output_rels_t), dim=0)

            batch_size = len(doc_ids)

            # OPT_PUB
            output_rels_h = sigmoid(output_rels_h)
            res_h = output_rels_h.detach().cpu().numpy()
            output_rels_t = sigmoid(output_rels_t)
            res_t = output_rels_t.detach().cpu().numpy()


            for i, doc_id in enumerate(doc_ids):
                title_vertex_id, title_vertex = title_entities[doc_id][0]

                doc = dat_file[doc_id]
                vertexSet = doc['vertexSet']
                result_d = result[doc_id]

                for eid, vertex in enumerate(vertexSet):
                    if eid == title_vertex_id:
                        continue

                    #dht_list.append((doc_id, title_vertex_id, eid))
                    conf = res_h[i, eid]
                    #conf_list.append(conf)
                    result_dict[(doc_id, title_vertex_id, eid)] = conf

                    #dht_list.append((doc_id, eid, title_vertex_id))
                    conf = res_t[i, eid]
                    #conf_list.append(conf)
                    result_dict[(doc_id, eid, title_vertex_id)] = conf



        dht_list = sorted(list(result_dict.keys()))
        conf_list = [result_dict[key] for key in dht_list]

        confidence = np.stack(conf_list)
        np.save(os.path.join(CONFIDENCE_DIR,f"confidence_{fname}_{data_opt}.npy"), confidence)
        json.dump(dht_list, open(os.path.join(CONFIDENCE_DIR,f"dht_{fname}_{data_opt}.json"),"w"))


    def full_test(self, model, input_theta = -1.0):
        result_title_ext = []

        dat_file = self.test_file
        get_batches = self.get_test_batch_np
        dat_len = self.test_len
        title_entities = self.test_title_entities

        sigmoid = nn.Sigmoid()

        model.eval()
        for data in get_batches():
            doc_ids = data['doc_ids']
            output_rels_h, output_rels_t = self.forward(model, data)

            batch_size = len(doc_ids)


            # OPT_PUB
            output_rels_h = sigmoid(output_rels_h)
            res_h = output_rels_h.detach().cpu().numpy()
            output_rels_t = sigmoid(output_rels_t)
            res_t = output_rels_t.detach().cpu().numpy()

            for i in range(batch_size):
                idx = doc_ids[i]
                doc = dat_file[idx]
                title_vertex_id, title_vertex = title_entities[idx][0]
                vertexSet = doc["vertexSet"]
                for eid, entity in enumerate(vertexSet):
                    for r in range(1, self.relation_num):
                        result_title_ext.append((res_h[i, eid, r].item(), idx, title_vertex_id, r, eid))
                        result_title_ext.append((res_t[i, eid, r].item(), idx, eid, r, title_vertex_id))

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
        get_batches = self.get_test_batch_np
        dat_len = self.test_len
        title_entities = self.test_title_entities

        softmax = nn.Softmax(dim=-1)
        sigmoid = nn.Sigmoid()


        model.eval()
        for data in get_batches():
            doc_ids = data['doc_ids']
            output_rels_h, output_rels_t = self.forward(model, data)

            batch_size = len(doc_ids)


            # OPT_PUB
            output_rels_h = sigmoid(output_rels_h)
            res_h = output_rels_h.detach().cpu().numpy()
            output_rels_t = sigmoid(output_rels_t)
            res_t = output_rels_t.detach().cpu().numpy()

            for i in range(batch_size):
                idx = doc_ids[i]
                doc = dat_file[idx]
                title_vertex_id, title_vertex = title_entities[idx][0]
                vertexSet = doc["vertexSet"]
                for eid, entity in enumerate(vertexSet):
                    for r in range(1, self.relation_num):
                        result_title_ext.append((res_h[i, eid, r].item(), idx, title_vertex_id, r, eid))
                        result_title_ext.append((res_t[i, eid, r].item(), idx, eid, r, title_vertex_id))

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


    def get_result(self, model, data_opt="test", theta=0.5, result_opt=RESULT_OPT_PUBLISH):
        if data_opt == "train":
            dat_file = self.train_file
            get_batches = self.get_train_batch_np
            dat_len = self.train_len
            title_entities = self.train_title_entities
            #dat_emid2eidmid = self.train_emid2eidmid
        elif data_opt == "test":
            dat_file = self.test_file
            get_batches = self.get_test_batch_np
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

            # output_rels = model(context_idxs, context_ner, context_char_idxs, title_map, em_map)
            # output_rels_h, output_rels_t = model(context_idxs, context_ner, context_char_idxs, title_map, em_map)
            output_rels_h, output_rels_t = self.forward(model, data)
            #output_rels = torch.cat((output_rels_h, output_rels_t), dim=0)

            batch_size = len(doc_ids)


            # OPT_PUB
            output_rels_h = sigmoid(output_rels_h)
            res_h = (output_rels_h >= theta).cpu().numpy()
            output_rels_t = sigmoid(output_rels_t)
            res_t = (output_rels_t >= theta).cpu().numpy()


            for i, doc_id in enumerate(doc_ids):
                title_vertex_id, title_vertex = title_entities[doc_id][0]

                doc = dat_file[doc_id]
                vertexSet = doc['vertexSet']
                result_d = result[doc_id]

                for eid, vertex in enumerate(vertexSet):
                    if eid == title_vertex_id:
                        continue

                    rh_list = res_h[i, eid].nonzero()[0].tolist()
                    rt_list = res_t[i, eid].nonzero()[0].tolist()
                    for r_h in rh_list:
                        if r_h > 0:
                            rec = (title_vertex_id, eid, r_h)
                            result_d.add(rec, -1)

                    for r_t in rt_list:
                        if r_t > 0:
                            rec = (eid, title_vertex_id, r_t)
                            result_d.add(rec, -1)
        return result

    def get_expname(self, model):
        option_dict = {
            "model_name": model.name,
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "train_prefix": self.train_prefix
        }
        return get_expname(option_dict)



    def get_loss(self, output, data):
        output_rels_h, output_rels_t = output
        output_rels = torch.cat((output_rels_h, output_rels_t), dim=0)

        ht_label = data['ht_label']
        label_mask = torch.cat((data['label_mask'], data['label_mask']), dim=0)

        loss_mat = self.bce_loss_logits(output_rels, ht_label)
        loss_mat = loss_mat * label_mask
        # output_r = output.view(-1, output.size(-1))
        # label_r = ht_labels.view(-1, output.size(-1))
        #
        # loss_mat = self.bce_loss_logits(output_r, label_r).view(output.shape)
        # loss = 0.0
        # for i in range(num_recs):
        #     loss = loss + loss_mat[i, :n_ems[i]].sum() + loss_mat[i + num_recs, :n_ems[i]].sum()
        loss = loss_mat.sum()/label_mask.sum()

        return loss

    def forward(self, model, data):
        context_idxs = data['context_idxs']
        context_ner = data['context_ner']
        context_char_idxs = data['context_char_idxs']
        title_map = data['title_map']
        em_map = data['em_map']
        #n_ems = data['n_ems']
        ht_label = data['ht_label']
        label_mask = data['label_mask']

        if "mention_cumsum" in data:
            mention_cumsum = data["mention_cumsum"]
            label_mask_m = data["label_mask_m"]
        else:
            mention_cumsum = None
            label_mask_m = None

        if self.use_bert:
            context_masks = data['context_masks']
            context_starts = data['context_starts']
            output_rels_h, output_rels_t = model(context_idxs, context_ner, context_char_idxs, title_map, em_map,
                                                 label_mask, label_mask_m, mention_cumsum,
                                                 context_masks = context_masks, context_starts = context_starts)
        else:
            output_rels_h, output_rels_t = model(context_idxs, context_ner, context_char_idxs, title_map, em_map, label_mask, label_mask_m, mention_cumsum)

        return output_rels_h, output_rels_t

    @conditional_profiler
    def train(self, model_pattern):
        model, optimizer = self.load_model_optimizer(model_pattern)
        exp_name = self.get_expname(model_pattern)

        if not self.save_name:
            self.save_name = exp_name

        dt = datetime.datetime.now()
        log_dir = f"{dt.day}_{dt.hour}:{dt.minute}_{exp_name}"
        if TENSORBOAD_LOADED:
            writer = SummaryWriter(log_dir=f"{WRITER_DIR_TITLE}/{log_dir}", comment=exp_name)
        else:
            writer = DummyWriter()



        for epoch in range(self.epoch, self.max_epoch):
            inst_count = 0
            t_begin = datetime.datetime.now()
            self.epoch = epoch
            model.train()
            loss_epoch = 0.0
            for data in self.get_train_batch_np():
                output = self.forward(model, data)
                loss = self.get_loss(output, data)
                # loss = loss + model.get_regularizer()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                inst_count+= 1


            loss_epoch /= self.train_batches
            t_end = datetime.datetime.now()
            print(
                f"Epoch {epoch}  Loss {loss_epoch} Time: {t_end - t_begin} TPB: {(t_end - t_begin) / inst_count}")
            writer.add_scalar("learning/loss_epoch", loss_epoch, epoch)


            if (epoch + 1) % self.test_epoch == 0 or epoch <1:
                model.eval()

                f1, precision, recall = self.test(model, writer = writer,  epoch = epoch, opt="test")
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_epoch = epoch
                    if not DEBUG_NOSAVE:
                        self.save_model(model, optimizer, True)
                else:
                    if epoch - self.best_epoch >=30:
                        print(f"Earlystop  Best epoch: {self.best_epoch}")
                        if not DEBUG_NOSAVE:
                            self.save_model(model, optimizer)
                        break

            if (epoch + 1) % self.checkpoint_epoch == 0 or (epoch+1) == self.max_epoch:
                if not DEBUG_NOSAVE:
                    self.save_model(model, optimizer)

        return model

    def set_entity_mention_map(self, i, vertexSet, title_entity_id, title_mapping, em_mapping, label_mask, label_mask_m, mention_offset):
        mentions = []

        num_title_mentions = len(vertexSet[title_entity_id])
        num_entities = len(vertexSet)
        for eid, vertex in enumerate(vertexSet):
            if eid == title_entity_id:
                for mid, em in enumerate(vertex):
                    em_pos_b, em_pos_e = em['pos']
                    title_mapping[i, mid, em_pos_b:em_pos_e] = 1.0 / (em_pos_e - em_pos_b)

                if not EM_INCLUDE_TITLE:
                    continue

            offset = mention_offset[eid]

            label_mask[i, eid] = 1.0
            label_mask_m[i,offset:offset + len(vertex)] = 1.0
            for mid, em in enumerate(vertex):
                em_pos_b, em_pos_e = em['pos']
                idx_mention = offset + mid

                em_mapping[i, idx_mention, em_pos_b:em_pos_e] = 1.0 / (em_pos_e - em_pos_b)
                mentions.append((eid, mid))

        return num_title_mentions, num_entities


    def _get_train_batch(self, b, train_order):
        start_id = b * self.batch_size
        cur_bsz = min(self.batch_size, self.train_len - start_id)
        cur_batch = list(train_order[start_id: start_id + cur_bsz])
        #cur_batch.sort(key=lambda x: self.train_num_entitymentions[x], reverse=True)
        max_length = self.max_length
        num_rels = len(self.rel2id)

        common_shape = (cur_bsz, max_length)
        shape_txt = common_shape
        context_idxs = np.zeros(shape_txt, dtype=np.int)
        context_ner = np.zeros(shape_txt, dtype=np.int)

        if self.use_bert:
            context_masks = np.zeros(shape_txt, dtype=np.int)
            context_starts = np.zeros(shape_txt, dtype=np.int)


        sent_lengths = []
        for i in cur_batch:
            if self.use_bert:
                len_w = np.where(self.data_train_bert_word[i] == 102)[0].item() + 1
            else:
                len_w = np.sum(self.data_train_word[i] > 0).item()
            sent_lengths.append(len_w)

        max_length = max(sent_lengths)

        context_char_idxs = np.zeros(common_shape + (self.char_limit,), dtype=np.int)

        title_mapping = np.zeros((cur_bsz, self.train_max_title_mentions, max_length))


        shape_emm = (cur_bsz, self.train_max_entities * self.train_max_mentions_per_entity, max_length)
        em_mapping = np.zeros(shape_emm)
        label_mask = np.zeros((cur_bsz, self.train_max_entities))
        label_mask_m = np.zeros(shape_emm[:-1] + (1,))

        common_shape_bc = (2 * cur_bsz, self.train_max_entities, num_rels)
        ht_label = np.zeros(common_shape_bc)

        max_entity_cnt = 0
        num_max_mentions = np.zeros(self.train_max_entities, dtype = np.int)
        for i, index in enumerate(cur_batch):
            ins = self.train_file[index]
            vertexSet = ins['vertexSet']
            num_entities = len(vertexSet)
            max_entity_cnt = max(max_entity_cnt, num_entities)
            for eid, mentions in enumerate(vertexSet):
                num_max_mentions[eid] = max(len(mentions),num_max_mentions[eid])

        mention_cumsum = num_max_mentions.cumsum()
        mention_offset = np.roll(mention_cumsum,1)
        mention_offset[0] = 0
        mention_size = mention_cumsum[-1]

        max_title_cnt = 0

        doc_ids = []
        for i, index in enumerate(cur_batch):
            ins = self.train_file[index]
            ins_single = self.train_file_single[index]
            doc_ids.append(index)

            doclen = ins['Ls'][-1]
            doclen = sent_lengths[i]
            if self.use_bert:
                context_idxs[i, :doclen] = self.data_train_bert_word[index, :doclen]
                context_masks[i, :doclen] = self.data_train_bert_mask[index, :doclen]
                context_starts[i, :doclen] = self.data_train_bert_starts[index, :doclen]
            else:
                context_idxs[i, :doclen] = self.data_train_word[index, :doclen]

            context_ner[i, :doclen] = self.data_train_ner[index, :doclen]
            context_char_idxs[i, :doclen] = self.data_train_char[index, :doclen]

            # for j in range(doclen):
            #	pos_idx[i, j] = j + 1

            title_vertex_id, title_vertex = self.train_title_entities[index][0]

            vertexSet = ins['vertexSet']

            num_title_mentions, num_entities = self.set_entity_mention_map(i, vertexSet, title_vertex_id, title_mapping, em_mapping, label_mask, label_mask_m, mention_offset)
            max_title_cnt = max(num_title_mentions, max_title_cnt)

            e2labels_h = self.train_e2labels_h[index]
            e2labels_t = self.train_e2labels_t[index]

            for eid, labels_h in enumerate(e2labels_h):
                for r in labels_h:
                    ht_label[i, eid, r] = 1
            for eid, labels_t in enumerate(e2labels_t):
                for r in labels_t:
                    ht_label[cur_bsz+i, eid, r] = 1

        input_lengths = (context_idxs[:cur_bsz] > 0).sum(axis=1)
        max_c_len = int(input_lengths.max())

        batch = {
            'torch': False,
            'b': b,
            'context_idxs': context_idxs[:cur_bsz, :max_c_len],
            'context_ner': context_ner[:cur_bsz, :max_c_len],
            'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len],
            'doc_ids': doc_ids,
            'title_map': title_mapping[:cur_bsz, :max_title_cnt, :max_c_len],
            'em_map': em_mapping[:cur_bsz, :mention_size, :max_c_len],
            #'n_ems': n_ems[:cur_bsz],
            'input_lengths': input_lengths[:cur_bsz],
            'ht_label': ht_label[:2 * cur_bsz, :max_entity_cnt],
            'label_mask': np.expand_dims(label_mask[:cur_bsz, :max_entity_cnt],-1),
            "mention_cumsum": mention_cumsum[:max_entity_cnt],
            "label_mask_m":label_mask_m[:cur_bsz, :mention_size],
        }

        if self.use_bert:
            batch["context_masks"] = context_masks[:cur_bsz, :max_c_len]
            batch["context_starts"] = context_starts[:cur_bsz, :max_c_len]

        return batch


    def test(self, model, writer=None, input_theta = 0.5, epoch=-1, opt="test"):
        if opt == "train":
            dat_file = self.train_file
            dat_file_single = self.train_file_single
            title_entities = self.train_title_entities
        elif opt == "test":
            dat_file = self.test_file
            dat_file_single = self.test_file_single
            title_entities = self.test_title_entities
        else:
            print("Errr")

        res = self.get_result(model, data_opt = opt, theta= input_theta)

        n_pos = 0
        n_labels = 0
        n_tp = 0
        for doc_id in range(len(dat_file)):
            doc = dat_file[doc_id]
            doc_single = dat_file_single[doc_id]

            title_vid, vertex = title_entities[doc_id][0]

            res_d = res[doc_id]
            n_pos += len(res_d)
            n_labels += len(doc_single['h_labels'])
            n_labels += len(doc_single['t_labels'])

            for label in doc_single['h_labels']:
                key = (label['h'], label['t'], label['r'])
                if key in res_d:
                    n_tp += 1

            for label in doc_single['t_labels']:
                key = (label['h'], label['t'], label['r'])
                if key in res_d:
                    n_tp += 1

        if n_tp == 0:
            f1 = precision = recall = 0.0
        else:
            precision = n_tp / n_pos
            recall = n_tp / n_labels
            f1 = 2 * precision * recall / (precision + recall)
        if writer is not None:
            writer.add_scalar(f"{opt}/n_pos", n_pos, epoch)
            writer.add_scalar(f"{opt}/precision", precision, epoch)
            writer.add_scalar(f"{opt}/recall", recall, epoch)
            writer.add_scalar(f"{opt}/f1", f1, epoch)

        print(f"{opt}  F1: {f1}, P: {precision}({n_tp}/{n_pos}), R: {recall} ({n_tp}/{n_labels})")
        return f1, precision, recall

    def _get_test_batch(self, b, test_order):
        start_id = b * self.batch_size
        cur_bsz = min(self.batch_size, self.test_len - start_id)
        cur_batch = list(test_order[start_id: start_id + cur_bsz])
        #cur_batch.sort(key=lambda x: self.test_num_entitymentions[x], reverse=True)
        max_length = self.max_length
        num_rels = len(self.rel2id)

        common_shape = (cur_bsz, max_length)
        context_idxs = np.zeros(common_shape, dtype=np.int)
        context_ner = np.zeros(common_shape, dtype=np.int)

        if self.use_bert:
            context_masks = np.zeros(common_shape, dtype=np.int)
            context_starts = np.zeros(common_shape, dtype=np.int)


        sent_lengths = []
        for i in cur_batch:
            if self.use_bert:
                len_w = np.where(self.data_test_bert_word[i] == 102)[0].item() + 1
            else:
                len_w = np.sum(self.data_test_word[i] > 0).item()
            sent_lengths.append(len_w)
        max_length = max(sent_lengths)

        context_char_idxs = np.zeros(common_shape + (self.char_limit,), dtype=np.int)

        title_mapping = np.zeros((cur_bsz, self.test_max_title_mentions, max_length))

        shape_emm = (cur_bsz, self.test_max_entities * self.test_max_mentions_per_entity, max_length)
        em_mapping = np.zeros(shape_emm)
        label_mask = np.zeros((cur_bsz, self.test_max_entities))
        label_mask_m = np.zeros(shape_emm[:-1] + (1,))

        common_shape_bc = (2 * cur_bsz, self.test_max_entities, num_rels)
        ht_label = np.zeros(common_shape_bc)

        max_entity_cnt = 0
        num_max_mentions = np.zeros(self.test_max_entities, dtype=np.int)
        for i, index in enumerate(cur_batch):
            ins = self.test_file[index]
            vertexSet = ins['vertexSet']
            num_entities = len(vertexSet)
            max_entity_cnt = max(max_entity_cnt, num_entities)
            for eid, mentions in enumerate(vertexSet):
                num_max_mentions[eid] = max(len(mentions), num_max_mentions[eid])

        mention_cumsum = num_max_mentions.cumsum()
        mention_offset = np.roll(mention_cumsum, 1)
        mention_offset[0] = 0
        mention_size = mention_cumsum[-1]

        max_title_cnt = 0

        doc_ids = []
        for i, index in enumerate(cur_batch):
            ins = self.test_file[index]
            ins_single = self.test_file_single[index]
            doc_ids.append(index)

            doclen = sent_lengths[i]

            if self.use_bert:
                context_idxs[i, :doclen] = self.data_test_bert_word[index, :doclen]
                context_masks[i, :doclen] = self.data_test_bert_mask[index, :doclen]
                context_starts[i, :doclen] = self.data_test_bert_starts[index, :doclen]
            else:
                context_idxs[i, :doclen] = self.data_test_word[index, :doclen]
            # context_pos[i, :doclen] = self.data_test_pos[index, :doclen]
            context_ner[i, :doclen] = self.data_test_ner[index, :doclen]
            context_char_idxs[i, :doclen] = self.data_test_char[index, :doclen]

            # for j in range(doclen):
            #	pos_idx[i, j] = j + 1

            title_vertex_id, title_vertex = self.test_title_entities[index][0]

            vertexSet = ins['vertexSet']
            num_title_mentions, num_entities = self.set_entity_mention_map(i, vertexSet, title_vertex_id, title_mapping, em_mapping, label_mask, label_mask_m, mention_offset)
            max_title_cnt = max(num_title_mentions, max_title_cnt)

            e2labels_h = self.test_e2labels_h[index]
            e2labels_t = self.test_e2labels_t[index]

            for eid, labels_h in enumerate(e2labels_h):
                for r in labels_h:
                    ht_label[i, eid, r] = 1
            for eid, labels_t in enumerate(e2labels_t):
                for r in labels_t:
                    ht_label[cur_bsz + i, eid, r] = 1

        input_lengths = (context_idxs[:cur_bsz] > 0).sum(axis=1)
        max_c_len = int(input_lengths.max())

        batch = {
            'torch': False,
            'b': b,
            'doc_ids': doc_ids,
            'context_idxs': context_idxs[:cur_bsz, :max_c_len],
            'context_ner': context_ner[:cur_bsz, :max_c_len],
            'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len],

            'title_map': title_mapping[:cur_bsz, :max_title_cnt, :max_c_len],
            'em_map': em_mapping[:cur_bsz, :mention_size, :max_c_len],
            #'n_ems': n_ems[:cur_bsz],

            'input_lengths': input_lengths[:cur_bsz],
            'ht_label': ht_label[:2 * cur_bsz, :max_entity_cnt],
            'label_mask': np.expand_dims(label_mask[:cur_bsz, :max_entity_cnt], -1),
            "mention_cumsum": mention_cumsum[:max_entity_cnt],
            "label_mask_m": label_mask_m[:cur_bsz, :mention_size],
        }
        if self.use_bert:
            batch["context_masks"] = context_masks[:cur_bsz, :max_c_len]
            batch["context_starts"] = context_starts[:cur_bsz, :max_c_len]

        return batch


