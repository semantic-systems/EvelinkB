# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
import json, pickle
from flair.data import Sentence
from flair.models import SequenceTagger
import flair
import os,io
import requests

from blink.biencoder.biencoder import BiEncoderRanker
from pytorch_transformers.tokenization_bert import BertTokenizer

import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser


from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG, TYPE_TAG_MAPPING

def GetWikiID(id2title, title):
	title = title.replace(" ", "_")
	if title in id2title[1]:
		return id2title[1][title]
	else:
		if title[0] == "'" and title[-1] == "'":
			title_form = title[1:-1]
			return id2title[1][title_form]
	return ''

def GetSubeventMention(data_sentence, tagger):
    all_events = []
    if 20000 > len(data_sentence) > 100:
        sentence = Sentence(data_sentence)
        # predict NER tags
        tagger.predict(sentence)
        # print predicted NER spans
        # print('The following events are found:
        # iterate over entities and print
        for entity in sentence.get_spans('ner'):
            if entity.tag == "EVENT" and entity.score > 0.7:
                all_events.append(entity.text)
    else:
        chunks, chunk_size = len(data_sentence), len(data_sentence) // 6
        sub_sentences = [data_sentence[i:i + chunk_size] for i in range(0, chunks, chunk_size)]
        for s in sub_sentences:
            sentence = Sentence(s)
            # predict NER tags
            tagger.predict(sentence)
            # print predicted NER spans
            # print('The following events are found:
            # iterate over entities and print
            for entity in sentence.get_spans('ner'):
                if entity.tag == "EVENT" and entity.score > 0.7:
                    all_events.append(entity.text)

    # removing duplicates
    all_events_unique = list(set(all_events))
    return all_events_unique


def get_context_representation(
        sample,
        tokenizer,
        max_seq_length,
        mention_key="mention",
        context_key="context",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
        # mention_tokens = [ent_start_token] + ['[MASK]'] + [ent_end_token]

    entity_tokens = []
    if 'entities' in sample and len(sample['entities']) > 0:
        for entity in sample['entities'][0:10]:
            tokens = tokenizer.tokenize(entity['form'])
            tokens = [TYPE_TAG_MAPPING[entity['type']][0]] + tokens + [TYPE_TAG_MAPPING[entity['type']][1]]
            # tokens = ["[SEP]"] + tokens
            entity_tokens += tokens

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens) - len(entity_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - len(entity_tokens) - left_quota - 2

    if left_quota <= 0 or right_quota <= 0:
        left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
        right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
        entity_tokens = []

    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
            context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + entity_tokens + ["[SEP]"]
    # print("context tokens", context_tokens)
    # print("entity tokens",entity_tokens)
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    if len(input_ids) != max_seq_length:
        print(max_seq_length, len(mention_tokens))
        print(left_quota, right_quota)
        print(mention_tokens)
        print(entity_tokens)
        print(len(entity_tokens))
        print(context_left)
        print(context_right)
        print(context_tokens)
        print(len(input_ids), max_seq_length)
    # else:
    #     print(len(input_ids), max_seq_length)
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
        candidate_desc,
        tokenizer,
        max_seq_length,
        sub_events=None,
        candidate_title=None,
        title_tag=ENT_TITLE_TAG,
        hyperlinks=None
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    title_tokens = []
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        title_tokens = title_tokens + [title_tag]

    link_tokens = []
    # print(hyperlinks)

    if hyperlinks is not None:
        for link in hyperlinks:
            tokens = tokenizer.tokenize(link[1])
            tokens = ["[SEP]"] + tokens
            link_tokens += tokens

    # print(link_tokens)
    sub_events_tokens = []

    # print(type(sub_events))
    # print("Sub Events:", sub_events)
    if sub_events is not None:
        if len(sub_events) > 10:
            sub_events = sub_events[0:10]
            # print("reduce sub events",sub_events)
            for sub_event in sub_events:
                tokens = tokenizer.tokenize(sub_event)
                tokens = ["[SEP]"] + tokens
                sub_events_tokens += tokens
        else:
            for sub_event in sub_events:
                tokens = tokenizer.tokenize(sub_event)
                tokens = ["[SEP]"] + tokens
                sub_events_tokens += tokens

    # print("Link Tokens", link_tokens)
    # print("Sub Event Token: ", sub_events_tokens)

    cand_tokens = cand_tokens[: max_seq_length - len(title_tokens) - len(link_tokens) - len(sub_events_tokens) - 2]
    cand_tokens = [cls_token] + title_tokens + cand_tokens + link_tokens + sub_events_tokens + [sep_token]

    # print("Cand Token: ", cand_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    if len(input_ids) != max_seq_length:
        print(title_tokens)
        print(len(title_tokens))
        print(link_tokens)
        print(len(link_tokens))
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }

def getCandidateData(tagger,title:str,id2title:dict,t2h:dict,title_text:dict):
    if title[0] == "'" and title[-1] == "'":
        title = title[1:-1]
    hyperlinks = []
    sub_events = []
    if title.replace(" ", '_') in id2title[1] and id2title[1][title.replace(" ", '_')] in t2h:
        for idx, hyp in enumerate(t2h[id2title[1][title.replace(" ", '_')]]):
            if idx > 10:
                break
            link = t2h[id2title[1][title.replace(" ", '_')]][hyp]
            if link['end'] > 2000:
                continue
            hyperlinks.append((hyp, title_text[title.replace("_", ' ')][link['start']: link['end']]))

    text = title_text[title.replace("_", ' ')].strip()
    try:
        sub_events = GetSubeventMention(text, tagger)
    except:
        print(f"No Sub Events Found For Wiki Title {title}")

    return text[0:2000],sub_events,title,hyperlinks


def process_mention_data(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        mention_key="mention",
        context_key="context",
        candidates_key='edl',
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
):
    text_file = open('data/wikipedia/wiki/title_text.json', 'r')
    title_text = json.load(text_file)
    print('Text loaded.')

    id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
    id2title = pickle.load(id2title_file)
    print('ID to title loaded.')

    t2hyperlinks_file = open('data/wikipedia/wiki/t2hyperlinks.json', 'r')
    t2h = json.load(t2hyperlinks_file)
    print('HyperLinks loaded.')

    flair.device = torch.device('cuda:1')
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    print('Sequence Tagger loaded.')


    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    all_context_tokens=[]
    all_candidate_tokens=[]
    all_label_tokens=[]
    for idx, sample in enumerate(iter_):

        all_candidates = sample[candidates_key]
        gold_id = sample['label_id']

        pointer = -1
        for j in range(0,len(all_candidates)):
            if id2title is not None:
                title = all_candidates[j]
                if title.replace(" ", "_") not in id2title[1]:
                    continue
                predict_id = GetWikiID(id2title, title)
                if int(predict_id) == int(gold_id):
                    pointer = j
                    ## print("Label Pointer: ", pointer, " Predicted ID: ", predict_id, " Gold ID: ", gold_id)
                    break
        if pointer == -1:
            continue

        sample_context_token = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        ## print("\n\n Context Token", sample_context_token["tokens"],"\n\n")

        sample_candidate_tokens=[]

        for candidate in all_candidates:
            desc, sub_events, title, hyperlinks = getCandidateData(tagger,title=candidate,id2title=id2title,t2h=t2h,title_text=title_text)
            ## if p == pointer:
            ##    print ("Desc: ",desc," Sub_events: ", sub_events," Title: ", title," Hyp: ", hyperlinks)
            if len(hyperlinks) > 0:
                if len(sub_events) > 0:
                    cand_tokens = get_candidate_representation(
                        desc, tokenizer, max_cand_length, sub_events=sub_events,candidate_title= title, hyperlinks=hyperlinks
                    )
                else:
                    cand_tokens = get_candidate_representation(
                        desc, tokenizer, max_cand_length, candidate_title=title, hyperlinks=hyperlinks
                    )
            else:
                if len(sub_events) > 0:
                    cand_tokens = get_candidate_representation(
                        desc, tokenizer, max_cand_length, sub_events=sub_events, candidate_title=title
                    )
                else:
                    cand_tokens = get_candidate_representation(
                        desc, tokenizer, max_cand_length, candidate_title=title
                    )
            ## if p == pointer:
            ##  print("\n\n Correct Candidate Token", cand_tokens["tokens"], "\n\n")
            sample_candidate_tokens.append(cand_tokens["ids"])

        all_context_tokens.append(sample_context_token["ids"])
        all_candidate_tokens.append(sample_candidate_tokens)
        all_label_tokens.append(pointer)

    nn_context = torch.LongTensor(all_context_tokens)
    nn_candidates = torch.LongTensor(all_candidate_tokens)
    nn_label = torch.LongTensor(all_label_tokens)

    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_label
    }

    try:
        save_data_path = os.path.join("./out/", "new_train_p3.t7")
        torch.save(nn_data, save_data_path)
    except:
        save_data_path = os.path.join("./out/", "new_train_exp_p3.json")
        outfile = open(save_data_path, 'w')
        json.dump(nn_data, outfile)

    return nn_data


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()

    params = args.__dict__

    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    train_samples=[]
    with io.open('./out/train_result_dp_top30.json', mode="r", encoding="utf-8") as file:
        for line in file:
            train_samples.append(json.loads(line.strip()))

    print("Length data_to_link",len(train_samples))
    train_samples = train_samples[30000:45000]

    data = process_mention_data(train_samples,tokenizer,256,256,False)

    print("Context Vectors: ", data["context_vecs"].size())
    print("Candidate Vectors: ", data["candidate_vecs"].size())
    print("Label Vectors: ", data["labels"].size())

