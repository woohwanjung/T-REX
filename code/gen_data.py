import numpy as np
import os
import json
import argparse
import random
from pytorch_transformers import *

from config import Config
from models.transformertokenizer import TransformerTokenizer
from tqdm import tqdm


from train import get_ext_parser

DEBUG = False




parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../data")
parser.add_argument('--out_path', type = str, default = "prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
sent_limit = 25
word_size = 100
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
train_ha_aug_file_name = os.path.join(in_path, 'train_ha_aug_ds.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])


if DEBUG:
    print("===========DEBUG - no save ==========")

bert_tok = TransformerTokenizer(BertModel, 'bert-base-uncased', "bert")
#transfoxl_tok = TransformerTokenizer(TransfoXLModel, 'transfo-xl-wt103')


def sents_2_idx(sents, word2id):
    #sents_idx = np.zeros([sent_limit, word_size]) + word2id['BLANK']
    sents_idx = []
    start_idx = 0
    for i, sent in enumerate(sents[:sent_limit]):
        sents_idx.append(list(range(start_idx, start_idx+len(sent))))
        start_idx += len(sent)
    return sents_idx

def init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):
    ori_data = json.load(open(data_file_name))


    Ma = 0
    Ma_e = 0
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0
    word2id = json.load(open(os.path.join(out_path, "word2id.json")))

    if DEBUG:
        iterator = range(1000)
    else:
        iterator = tqdm(range(len(ori_data)))

    for i in iterator:
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        vertexSet =  ori_data[i]['vertexSet']
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        new_labels = []
        for label in labels:
            rel = label['r']
            assert(rel in rel2id)
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))


            if suffix=='_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))


            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                # fix a bug here
                label['intrain'] = False
                label['indev_train'] = False

                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                label['indev_train'] = True


            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))

        item['na_triple'] = na_triple
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        item['sents_idx'] = sents_2_idx(ori_data[i]['sents'], word2id)
        data.append(item)

        Ma = max(Ma, len(vertexSet))
        Ma_e = max(Ma_e, len(item['labels']))


    print ('data_len:', len(ori_data))
    # print ('Ma_V', Ma)
    # print ('Ma_e', Ma_e)
    # print (suffix)
    # print ('fact_in_train', len(fact_in_train))
    # print (intrain, notintrain)
    # print ('fact_in_devtrain', len(fact_in_dev_train))
    # print (indevtrain, notindevtrain)


    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    if not DEBUG:
        json.dump(data , open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    # id2char= {v:k for k,v in char2id.items()}
    # json.dump(id2char, open("data/id2char.json", "w"))

    word2id = json.load(open(os.path.join(out_path, "word2id.json")))
    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)

    bert_token = np.zeros((sen_tot, max_length), dtype = np.int64)
    bert_mask = np.zeros((sen_tot, max_length), dtype = np.int64)
    bert_starts = np.zeros((sen_tot, max_length), dtype = np.int64)

    '''
    transfoxl_token = np.zeros((sen_tot, max_length), dtype = np.int64)
    transfoxl_mask = np.zeros((sen_tot, max_length), dtype = np.int64)
    transfoxl_starts = np.zeros((sen_tot, max_length), dtype = np.int64)
    '''

    if DEBUG:
        iterator = range(len(ori_data))
    else:
        iterator = tqdm(range(len(ori_data)))
    for i in iterator:
        item = ori_data[i]
        words = []
        for sent in item['sents']:
            words += sent

        bert_token[i], bert_mask[i], bert_starts[i] = bert_tok.subword_tokenize_to_ids(words)
        #transfoxl_token[i], transfoxl_mask[i], transfoxl_starts[i] = transfoxl_tok.subword_tokenize_to_ids(words)

        for j, word in enumerate(words):
            word = word.lower()

            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']

            for c_idx, k in enumerate(list(word)):
                if c_idx>=char_limit:
                    break
                sen_char[i,j,c_idx] = char2id.get(k, char2id['UNK'])

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

    print("Finishing processing")
    if not DEBUG:
        np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word)
        np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'), sen_pos)
        np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner)
        np.save(os.path.join(out_path, name_prefix + suffix + '_char.npy'), sen_char)
        np.save(os.path.join(out_path, name_prefix + suffix + '_bert_word.npy'), bert_token)
        np.save(os.path.join(out_path, name_prefix + suffix + '_bert_mask.npy'), bert_mask)
        np.save(os.path.join(out_path, name_prefix + suffix + '_bert_starts.npy'), bert_starts)

    print("Finish saving")



#init(train_ha_aug_file_name, rel2id, max_length = 512, is_training = False, suffix='_train_aug')

def split_indexes(dat_len, seed = 0):
    random.seed(seed)
    list_all = list(range(dat_len))
    list1 = random.sample(list_all, dat_len // 2)
    list1.sort()

    list2 = []
    i1 = 0
    for i in range(dat_len):
        if i1 < len(list1) and list1[i1] == list_all[i]:
            i1 += 1
        else:
            list2.append(list_all[i])

    assert  len(list_all) == (len(list1)+len(list2))
    print(list1[:10])
    print(list2[:10])

    return list1, list2


def split_npy(dat, idx_list1, idx_list2,  dim = 0):
    if dim == 0:
        tmp_dat = dat
    else:
        tmp_dat = np.swapaxes(dat, (0,dim))

    list1 = [tmp_dat[i] for i in idx_list1]
    list2 = [tmp_dat[i] for i in idx_list2]

    dat1 = np.stack(list1, axis = dim)
    dat2 = np.stack(list2, axis = dim)

    return dat1, dat2

def split_save_json(dat, idx_list1, idx_list2):
    dat1 = [dat[i] for i in idx_list1]
    dat2 = [dat[i] for i in idx_list2]

    out_path = "prepro_data"
    name_prefix = "dev_dev"
    path1 = os.path.join(out_path, f"{name_prefix}1.json")
    path2 = os.path.join(out_path, f"{name_prefix}2.json")

    json.dump(dat1,open(path1,"w"))
    json.dump(dat2,open(path2,"w"))

    return dat1, dat2



def save_numpy(dat_npy1, dat_npy2, name):
    out_path = "prepro_data"
    name_prefix = "dev_dev"
    path1 = os.path.join(out_path, f"{name_prefix}1_{name}.npy")
    path2 = os.path.join(out_path, f"{name_prefix}2_{name}.npy")

    np.save(path1, dat_npy1)
    np.save(path2, dat_npy2)




def split_dev_data():
    parser = get_ext_parser()
    args = parser.parse_args()
    config = Config(args)
    config.load_test_data()

    dev_len = config.test_len
    idx_list1, idx_list2 = split_indexes(dev_len)

    print(" ")
    split_save_json(config.test_file, idx_list1, idx_list2)

    data_word1, data_word2 = split_npy(config.data_test_word, idx_list1, idx_list2)
    save_numpy(data_word1,data_word2,"word")

    data_ner1, data_ner2 = split_npy(config.data_test_ner, idx_list1, idx_list2)
    save_numpy(data_ner1, data_ner2, "ner")

    data_pos1, data_pos2 = split_npy(config.data_test_pos, idx_list1, idx_list2)
    save_numpy(data_pos1, data_pos2, "pos")

    data_char1, data_char2 = split_npy(config.data_test_char, idx_list1, idx_list2)
    save_numpy(data_char1, data_char2, "char")

    data_bert_word1, data_bert_word2 = split_npy(config.data_test_bert_word, idx_list1, idx_list2)
    save_numpy(data_bert_word1, data_bert_word2, "bert_word")

    data_bert_mask1, data_bert_mask2 = split_npy(config.data_test_bert_mask, idx_list1, idx_list2)
    save_numpy(data_bert_mask1, data_bert_mask2, "bert_mask")

    data_bert_starts1, data_bert_starts2 = split_npy(config.data_test_bert_starts, idx_list1, idx_list2)
    save_numpy(data_bert_starts1, data_bert_starts2, "bert_starts")



if __name__ == "__main__":
    #'''
    init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
    init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
    init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
    init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')
    #'''
    split_dev_data()





