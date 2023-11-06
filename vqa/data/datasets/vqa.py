import os
import json
import _pickle as cPickle
from PIL import Image
import re
import base64
import numpy as np
import csv
import sys
import time
import pprint
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

from pycocotools.coco import COCO

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

class VQA(Dataset):
    def __init__(self, image_set, root_path, data_path, answer_vocab_file, use_imdb=True,
                 with_precomputed_visual_feat=False, boxes="36",
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=True, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, lang='en', data='en', **kwargs):
        """
        Visual Question Answering Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param lang: model use language
        :param dataset: 有四種en、FM-IQA、zh_vqa、demo
        :param kwargs:
        """
        super(VQA, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                      'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']

        if data == 'en':
            ##EN
            vqa_question = {
                "train2014": "vqa/HieQT_train_questions.json",
                "valminusminival2014": "vqa/v2_OpenEnded_mscoco_valminusminival2014_questions.json",
                "val2014": "vqa/HieQT_val_questions.json",
                "minival2014": "vqa/v2_OpenEnded_mscoco_minival2014_questions.json",
                "test-dev2015": "vqa/HieQT_test-dev_questions.json",
                "test2015": "vqa/HieQT_test_questions.json",
            }        
        
            vqa_annot = {
                "train2014": "vqa/v2_mscoco_train2014_annotations.json",
                "valminusminival2014": "vqa/v2_mscoco_valminusminival2014_annotations.json",
                "val2014": "vqa/v2_mscoco_val2014_annotations.json",
                "minival2014": "vqa/v2_mscoco_minival2014_annotations.json",
             }
            ##EN
        elif data == 'FM-IQA':
            #FM-IQA
            vqa_question = {
                "train2014": "vqa/FM-ZH-train_questions.json",
                "valminusminival2014": "vqa/v2_OpenEnded_mscoco_valminusminival2014_questions.json",
                "val2014": "vqa/FM-ZH-val_questions.json",
                "minival2014": "vqa/v2_OpenEnded_mscoco_minival2014_questions.json",
                "test-dev2015": "vqa/FM-ZH-test_questions.json",
                "test2015": "vqa/FM-ZH-test_questions.json",
                }
        
            vqa_annot = {
                "train2014": "vqa/FM-ZH-train_annotations.json",
                "valminusminival2014": "vqa/v2_mscoco_valminusminival2014_annotations.json",
                "val2014": "vqa/FM-ZH-val_annotations.json",
                "minival2014": "vqa/v2_mscoco_minival2014_annotations.json",
                }
            #FM-IQA
        elif data == 'zh_vqa':
            ##zh_vqa
            vqa_question = {
                    "train2014": "vqa/HieQTzh_train_questions.json",
                    "valminusminival2014": "vqa/v2_OpenEnded_mscoco_valminusminival2014_questions.json",
                    "val2014": "vqa/HieQTzh_val_questions.json",
                    "minival2014": "vqa/v2_OpenEnded_mscoco_minival2014_questions.json",
                    "test-dev2015": "vqa/HieQTzh_test_questions.json",
                    "test2015": "vqa/HieQTzh_test_questions.json",
                    }
            vqa_annot = {
                    "train2014": "vqa/zh_v2_mscoco_train2014_annotations.json",
                    "valminusminival2014": "vqa/v2_mscoco_valminusminival2014_annotations.json",
                    "val2014": "vqa/zh_vqav2_val_annotations.json",
                    "minival2014": "vqa/v2_mscoco_minival2014_annotations.json",
                    }
            ##zh_vqa
        elif data == 'demo':
            if lang == 'zh':
                vqa_question = {
                        "demo_img": "vqa/zh_demo_question.json",
                        }
            elif lang == 'en':
                vqa_question = {
                        "demo_img": "vqa/demo_question.json",
                        }
        
        vqa_imdb = {
            "train2014": "vqa/vqa_imdb/imdb_train2014.npy",
            "val2014": "vqa/vqa_imdb/imdb_val2014.npy",
            'test2015': "vqa/vqa_imdb/imdb_test2015.npy",
            'minival2014': "vqa/vqa_imdb/imdb_minival2014.npy",
            'test-dev2015': "vqa/vqa_imdb/imdb_test-dev2015.npy",
            'demo_img': "vqa/vqa_imdb/imdb_demo.npy"
        }

        if boxes == "36":
            precomputed_boxes = {
                'train2014': ("vgbua_res101_precomputed", "trainval_resnet101_faster_rcnn_genome_36"),
                "valminusminival2014": ("vgbua_res101_precomputed", "trainval_resnet101_faster_rcnn_genome_36"),
                'val2014': ("vgbua_res101_precomputed", "trainval_resnet101_faster_rcnn_genome_36"),
                "minival2014": ("vgbua_res101_precomputed", "trainval_resnet101_faster_rcnn_genome_36"),
                "test-dev2015": ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome_36"),
                "test2015": ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome_36"),
            }
        elif boxes == "10-100ada":
            if data == 'en' or data == 'demo':
                precomputed_boxes = {
                    'train2014': ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    "valminusminival2014": ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    'val2014': ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    "minival2014": ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    ##EN
                    "test-dev2015": ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome"),
                    "test2015": ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome"),
                    "demo_img": ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome"),
                    }
            elif data == 'FM-IQA' or data == 'zh_vqa':
                precomputed_boxes = {
                    'train2014': ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    "valminusminival2014": ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    'val2014': ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    "minival2014": ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    ##ZH
                    "test-dev2015": ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    "test2015": ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                    }
        else:
            raise ValueError("Not support boxes: {}!".format(boxes))
        if data == 'en' or data == 'demo':
            coco_dataset = {
                "train2014": ("train2014", "annotations/instances_train2014.json"),
                "valminusminival2014": ("val2014", "annotations/instances_val2014.json"),
                "val2014": ("val2014", "annotations/instances_val2014.json"),
                "minival2014": ("val2014", "annotations/instances_val2014.json"),
                ## EN
                "test-dev2015": ("test2015", "annotations/image_info_test2015.json"),
                "test2015": ("test2015", "annotations/image_info_test2015.json"),
                "demo_img": ("test2015", "annotations/image_info_test2015.json", "img"),
                }
        elif data == 'FM-IQA' or data == 'zh_vqa': ###
            coco_dataset = {
                "train2014": ("train2014", "annotations/instances_train2014.json"),
                "valminusminival2014": ("val2014", "annotations/instances_val2014.json"),
                "val2014": ("val2014", "annotations/instances_val2014.json"),
                "minival2014": ("val2014", "annotations/instances_val2014.json"),
                ##ZH
                "test-dev2015": ("test2015", "annotations/instances_val2014.json"),
                "test2015": ("test2015", "annotations/instances_val2014.json"),
                }

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!',
                      '。','，','、','；','：','？','『',
                      '』','「','」','）','（','！','．',
                      '《','》','〈','〉', '&']

        self.use_imdb = use_imdb
        self.boxes = boxes
        self.test_mode = test_mode
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_path = data_path
        self.root_path = root_path
        with open(answer_vocab_file, 'r', encoding='utf8') as f:
            self.answer_vocab = [w.lower().strip().strip('\r').strip('\n').strip('\r') for w in f.readlines()]
            self.answer_vocab = list(filter(lambda x: x != '', self.answer_vocab))
            if not self.use_imdb:
                self.answer_vocab = [self.processPunctuation(w) for w in self.answer_vocab]
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.ann_files = [os.path.join(data_path, vqa_annot[iset]) for iset in self.image_sets] \
            if not self.test_mode else [None for iset in self.image_sets]
        if data == 'demo':
            demo_path = os.path.join(root_path, 'demo/static/uploads/')
            self.q_files = [os.path.join(demo_path, vqa_question[iset]) for iset in self.image_sets]
        else:
            self.q_files = [os.path.join(data_path, vqa_question[iset]) for iset in self.image_sets]
        self.imdb_files = [os.path.join(data_path, vqa_imdb[iset]) for iset in self.image_sets]
        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset][0],
                         '{0}.zip@/{0}'.format(precomputed_boxes[iset][1])
                         if zip_mode else precomputed_boxes[iset][1])
            for iset in self.image_sets]
        self.box_bank = {}
        if data == 'demo':
            self.coco_datasets = [(os.path.join(demo_path,
                                                coco_dataset[iset][2],
                                                'COCO_{}_{{:012d}}.jpg'.format(coco_dataset[iset][0]))
                                   if not zip_mode else
                                   os.path.join(data_path,
                                                coco_dataset[iset][0] + '.zip@/' + coco_dataset[iset][0],
                                                'COCO_{}_{{:012d}}.jpg'.format(coco_dataset[iset][0])),
                                   os.path.join(data_path, coco_dataset[iset][1]))
                                   for iset in self.image_sets]
        else:
            self.coco_datasets = [(os.path.join(data_path,
                                                coco_dataset[iset][0],
                                                'COCO_{}_{{:012d}}.jpg'.format(coco_dataset[iset][0]))
                                   if not zip_mode else
                                   os.path.join(data_path,
                                                coco_dataset[iset][0] + '.zip@/' + coco_dataset[iset][0],
                                                'COCO_{}_{{:012d}}.jpg'.format(coco_dataset[iset][0])),
                                   os.path.join(data_path, coco_dataset[iset][1]))
                                 for iset in self.image_sets]
        self.transform = transform
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        if data == 'demo':
            self.cache_db = False
        else:
            self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        #print('-------------------data-----------------')
        #print(self.database)
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

    def convert_QType_to_ids(self, Q_type, num):
        Q_list = [0] * num
        if num == 7:
            if Q_type == 'YN':
                Q_list[0] = 1
            elif Q_type == 'ENTY':
                Q_list[1] = 1
                Q_list[5] = 1
            elif Q_type == 'NUM':
                Q_list[2] = 1
            elif Q_type == 'LOC':
                Q_list[3] = 1
            elif Q_type == 'HUM':
                Q_list[4] = 1
            elif Q_type == 'DESC':
                Q_list[5] = 1
                Q_list[1] = 1
            elif Q_type == 'ABBR':
                Q_list[6] = 1
        elif num == 3:
            if Q_type == 'yes/no':
                Q_list[0] = 1
            elif Q_type == 'number':
                Q_list[1] = 1
            elif Q_type == 'other':
                Q_list[2] = 1
                #Q_list = [0, 0]
            else:
                print('Q_type:', Q_type)
                print(doidjoid)
        elif num == 9:
            #Q_list = [0, 0, 0]
            if Q_type == 'yes/no':
                Q_list[0] = 1
                #Q_list = [0, 0, 1]
            elif Q_type == 'number':
                Q_list[1] = 1
                #Q_list = [0, 1, 0]
            elif Q_type == 'loc':
                Q_list[2] = 1
                #Q_list = [0, 1, 1]
            elif Q_type == 'human':
                Q_list[3] = 1
                #Q_list = [0, 0, 0]
            elif Q_type == 'color':
                Q_list[4] = 1
                #Q_list = [1, 0, 0]
            elif Q_type == 'sport':
                Q_list[5] = 1
                #Q_list = [1, 0, 1]
            elif Q_type == 'animal':
                Q_list[6] = 1
                #Q_list = [1, 1, 0]
            elif Q_type == 'brand':
                Q_list[7] = 1
                #Q_list = [1, 1, 1]
            elif Q_type == 'other':
                Q_list[8] = 1
                #Q_list = [0, 0, 0]
        else:
            print('QType Error')
            print(dhoid)
        return Q_list

    def get_mixword_list(self, s):
        #regEx = re.compile('[\\W]+')  # 我們可以使用正則表達式來切分句子，切分的規則是除單詞，數字外的任意字符串
        regEx = re.compile('\s+')
        res = re.compile(r"([\u4e00-\u9fa5])")  # [\u4e00-\u9fa5]中文範圍
        p1 = regEx.split(s.lower())
        str_list = []
        for partial_str in p1:
            if res.split(partial_str) == None:
                str_list.append(partial_str)
            else:
                ret = res.split(partial_str)
                for ch in ret:
                    str_list.append(ch)
        list_word = [w for w in str_list if len(w.strip()) > 0] # 去掉為空的字符
        return list_word

    #def extendTokenidWeight(self, seq, seq_weights, lang):
    def extendTokenidWeight(self, seq, seq_weights):
        tokens = []
        extend_weight = []
        ids = []
        for i, ps in enumerate(seq):
            partial_tokens = []
            partial_ids = []
            partial_weight = []
            partial_ps = self.get_mixword_list(ps)
            '''
            if lang == 'en':
                test_partial_ps = ps.split(' ')
            elif lang == 'zh':
                partial_ps = get_mixword_list(ps)
            '''
            
            for j, pps in enumerate(partial_ps):
                p_token = self.tokenizer.tokenize(pps)
                p_ids = self.tokenizer.convert_tokens_to_ids(p_token)
                partial_tokens.extend(p_token)
                partial_ids.extend(p_ids)
                try:
                    if len(p_token) > 1:
                        partial_weight.extend([seq_weights[i][j]] * len(p_token))
                    else:
                        partial_weight.extend([seq_weights[i][j]])
                except:
                    print('ps:', ps)
                    print('partial_ps:', partial_ps)
                    print('pps:', pps)
                    print('p_token:', p_token, len(p_token))
                    print('partial_tokens:', partial_tokens[i])
                    print('partial_ids:', partial_ids[i])
                    print('partial_weight:', partial_weight[i])
                    print('partial_weight:', partial_weight)
                    
            tokens.append(partial_tokens)
            extend_weight.append(partial_weight)
            ids.append(partial_ids)
        return tokens, extend_weight, ids

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'question', 'Q_type', 'word', 'word_weight', 'phrase', 'phrase_weight']
            #return ['image', 'boxes', 'im_info', 'question']
        else:
            return ['image', 'boxes', 'im_info', 'question', 'Q_type', 'word', 'word_weight', 'phrase', 'phrase_weight', 'label']
            #return ['image', 'boxes', 'im_info', 'question', 'label']

    def __getitem__(self, index):
        idb = self.database[index]

        # image, boxes, im_info
        boxes_data = self._load_json(idb['box_fn'])
        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = idb['width'], idb['height']

            boxes_features = torch.as_tensor(np.array(
                    np.frombuffer(self.b64_decode(boxes_data['features']), dtype=np.float32).reshape((boxes_data['num_boxes'], -1))))
        
        else:
            print('False')
            image = self._load_image(idb['image_fn'])
            w0, h0 = image.size
        boxes = torch.as_tensor(np.array(np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape((boxes_data['num_boxes'], -1))))

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                if 'image_box_feature' in boxes_data:
                    image_box_feature = torch.as_tensor(
                        np.frombuffer(
                            self.b64_decode(boxes_data['image_box_feature']), dtype=np.float32
                        ).reshape((1, -1))
                    )
                else:
                    image_box_feature = boxes_features.mean(0, keepdim=True)
                boxes_features = torch.cat((image_box_feature, boxes_features), dim=0)
        im_info = torch.tensor([w0, h0, 1.0, 1.0])
        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # flip: 'left' -> 'right', 'right' -> 'left'
        if self.use_imdb:
            q_tokens = idb['question_tokens']
        else:
            q_tokens = self.tokenizer.tokenize(idb['question'])
        if flipped:
            q_tokens = self.flip_tokens(q_tokens, verbose=False)
        
        if not self.test_mode:
            answers = idb['answers']
            if flipped:
                answers_tokens = [a.split(' ') for a in answers]
                answers_tokens = [self.flip_tokens(a_toks, verbose=False) for a_toks in answers_tokens]
                answers = [' '.join(a_toks) for a_toks in answers_tokens]
            answers = [self.processPunctuation(w) for w in answers] #new
            label = self.get_soft_target(answers)

        # question
        if self.use_imdb:
            q_str = ' '.join(q_tokens)
            q_retokens = self.tokenizer.tokenize(q_str)
        else:
            q_retokens = q_tokens
        q_ids = self.tokenizer.convert_tokens_to_ids(q_retokens)

        # concat box feature to box
        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=-1)

        #Q_type
        type_num = 9 #有3、7、9三種，分別表示vqav2 ans_type 、 TREC和vqav2擴充版分類type
        Q_type = idb['Q_type']
        Q_type = self.convert_QType_to_ids(Q_type, type_num)

        # Hie
        word = idb['word']
        word_weights = idb['word_weight']
        phrase = idb['phrase']
        phrase_weights = idb['phrase_weight']

        if len(word) != 0:
            #word_tokens, word_weights, word_ids = self.extendTokenidWeight(word, word_weights, lang)
            word_tokens, word_weights, word_ids = self.extendTokenidWeight(word, word_weights)
        else:
            word_tokens = []
            word_ids = []

        if len(phrase) != 0:
            #phrase_tokens, phrase_weights, phrase_ids = self.extendTokenidWeight(phrase, phrase_weights, lang)
            phrase_tokens, phrase_weights, phrase_ids = self.extendTokenidWeight(phrase, phrase_weights)
        else:
            phrase_tokens = []
            phrase_ids = []
        '''
        print('****** word ******')
        print(word)
        print('****** word_weight ******')
        print(word_weights)
        print('****** phrase ******')
        print(phrase)
        print('****** phrase_weight ******')
        print(phrase_weights)
        print('****** question_id *****')
        print(idb['question_id'])
        print('******q_ids******')
        print(idb['question'], ' : ', q_retokens, ' ', q_ids)
        print('******Q_type******')
        print(Q_type)
        
        print('****** answers *********')
        print(answers)
        print('****** label ***********')
        print(label.argmax().detach().cpu().tolist())
        print('------------------------END---------------------')
        print(djoidjdj)
        '''
        
        if self.test_mode:
            #return image, boxes, im_info, q_ids
            return image, boxes, im_info, q_ids, Q_type, word_ids, word_weights, phrase_ids, phrase_weights
        else:
            #print([(self.answer_vocab[i], p.item()) for i, p in enumerate(label) if p.item() != 0])
            return image, boxes, im_info, q_ids, Q_type, word_ids, word_weights, phrase_ids, phrase_weights, label
            #return image, boxes, im_info, q_ids, Q_type, label

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def answer_to_ind(self, answer):
        if answer in self.answer_vocab:
            return self.answer_vocab.index(answer)
        else:
            return self.answer_vocab.index('<unk>')

    def get_soft_target(self, answers):
        soft_target = torch.zeros(len(self.answer_vocab), dtype=torch.float)
        answer_indices = [self.answer_to_ind(answer) for answer in answers]
        gt_answers = list(enumerate(answer_indices))
        unique_answers = set(answer_indices)

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]

                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)

            if answer != self.answer_vocab.index('<unk>'):
                soft_target[answer] = avg_acc

        return soft_target

    def processPunctuation(self, inText):

        if inText == '<unk>':
            return inText

        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText

    def load_annotations(self):
        tic = time.time()
        database = []
        if self.use_imdb:
            db_cache_name = 'vqa2_imdb_boxes{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        else:
            db_cache_name = 'vqa2_nonimdb_boxes{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        if self.with_precomputed_visual_feat:
            db_cache_name += 'visualprecomp'
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipmode'
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))

        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        print('loading database of split {}...'.format('+'.join(self.image_sets)))
        tic = time.time()

        if self.use_imdb:
            for imdb_file, (coco_path, coco_annot), box_file \
                    in zip(self.imdb_files, self.coco_datasets, self.precomputed_box_files):
                print('-------------------------------------------imbd--------------------------------------')
                print("loading imdb: {}".format(imdb_file))
                imdb = np.load(imdb_file, allow_pickle=True)
                print("imdb info:")
                pprint.pprint(imdb[0])

                coco = COCO(coco_annot)
                for item in imdb[1:]:
                    print('item:', item)
                    idb = {'image_id': item['image_id'],
                           'image_fn': coco_path.format(item['image_id']),
                           'width': coco.imgs[item['image_id']]['width'],
                           'height': coco.imgs[item['image_id']]['height'],
                           'box_fn': os.path.join(box_file, '{}.json'.format(item['image_id'])),
                           'question_id': item['question_id'],
                           'question_tokens': item['question_tokens'],
                           'answers': item['answers'] if not self.test_mode else None,
                           }
                    database.append(idb)
        else:
            for ann_file, q_file, (coco_path, coco_annot), box_file \
                    in zip(self.ann_files, self.q_files, self.coco_datasets, self.precomputed_box_files):
                qs = self._load_json(q_file)['questions']
                anns = self._load_json(ann_file)['annotations'] if not self.test_mode else ([None] * len(qs))
                coco = COCO(coco_annot)
                for ann, q in zip(anns, qs):
                    idb = {'image_id': q['image_id'],
                           'image_fn': coco_path.format(q['image_id']),
                           'width': coco.imgs[q['image_id']]['width'],
                           'height': coco.imgs[q['image_id']]['height'],
                           'box_fn': os.path.join(box_file, '{}.json'.format(q['image_id'])),
                           'question_id': q['question_id'],
                           'question': q['question'],
                           'Q_type': q['question_type'],
                           'word': q['word'],
                           'word_weight' : q['word_weight'],
                           'phrase': q['phrase'],
                           'phrase_weight': q['phrase_weight'],
                           'answers': [a['answer'] for a in ann['answers']] if not self.test_mode else None,
                           'multiple_choice_answer': ann['multiple_choice_answer'] if not self.test_mode else None,
                           "question_type": ann['question_type'] if not self.test_mode else None,
                           "answer_type": ann['answer_type'] if not self.test_mode else None,
                           }
                    database.append(idb)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))
        return database
    
    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def load_precomputed_boxes(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = {}
            with open(box_file, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    item['image_id'] = int(item['image_id'])
                    item['image_h'] = int(item['image_h'])
                    item['image_w'] = int(item['image_w'])
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in (['boxes', 'features'] if self.with_precomputed_visual_feat else ['boxes']):
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode()),
                                                    dtype=np.float32).reshape((item['num_boxes'], -1))
                    in_data[item['image_id']] = item
            self.box_bank[box_file] = in_data
            return in_data

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

