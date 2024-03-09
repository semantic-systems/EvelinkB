# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import sys
import re
import dateutil.parser as dparser
import numpy as np
import wikipediaapi

from tqdm import tqdm
import blink.biencoder.data_process as data
from blink.common.params import ENT_START_TAG, ENT_END_TAG
from flair.data import Sentence
from flair.models import SequenceTagger

wiki_wiki = wikipediaapi.Wikipedia('EvelinkBFinal (evelinkb@gmail.com)', 'en')

def GetDate(title: str):
    if re.search('\d{2}th', title) or re.search('\d{2}nd', title):
        return None
    if re.search('no.\d{2}', title) or re.search(' \d{2} ', title):
        return None
    try:
        year = dparser.parse(title, fuzzy=True).year
        if 2024 > year > 1000:
            return year
    except:
        return None


def GetAllSections(sections, level=0):
    section_titles = []
    for s in sections:
        # print("%s: %s" % ("*" * (level + 1), s.title))
        section_titles.append(s.title)
        section_titles.extend(GetAllSections(s.sections, level + 1))
    return section_titles


def GetSectionsFromWiki(title: str):
    try:
        wp = wiki_wiki.page(title)
        sections = wp.sections
        return sections
    except:
        return None

def GetSubeventMention(data_sentence, tagger):
    all_events = []
    if 100 < len(data_sentence) < 25000:
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
        chunks, chunk_size = len(data_sentence), len(data_sentence) // 4
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

def prepare_crossencoder_mentions(
    tokenizer,
    samples,
    max_context_length=256,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):

    context_input_list = []  # samples X 128

    for sample in tqdm(samples):
        context_tokens = data.get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )
        tokens_ids = context_tokens["ids"]
        context_input_list.append(tokens_ids)

    context_input_list = np.asarray(context_input_list)
    return context_input_list


def prepare_crossencoder_candidates(
    tokenizer, labels, nns, id2title, id2text, id2hyper, max_cand_length=256, topk=200
):
    # tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    # print('Sequence Tagger loaded.')

    START_TOKEN = tokenizer.cls_token
    END_TOKEN = tokenizer.sep_token

    candidate_input_list = []  # samples X topk=10 X 128
    label_input_list = []  # samples
    idx = 0
    for label, nn in zip(labels, nns):
        candidates = []

        label_id = -1
        for jdx, candidate_id in enumerate(nn[:topk]):

            if label == candidate_id:
                label_id = jdx
            # print(id2text.keys())
            sub_event = []
            # try:
            #     sub_event = GetSubeventMention(id2text[candidate_id], tagger)
            # except Exception as e:
            #     print("No sub events for Candidate id: ", candidate_id, "Candidate title: ", id2title[candidate_id])
            #
            # year = GetDate(id2title[candidate_id].replace("_", ' ').replace("'", '').strip('\"'))
            year = None

            # sections = GetSectionsFromWiki(id2title[candidate_id].replace("_", ' '))
            # all_sections = []
            # if sections is not None:
            #     all_sections = GetAllSections(sections)

            overlap_sections = []
            # for item in sub_event:
            #     if item in all_sections:
            #         overlap_sections.append(item)

            if candidate_id in id2hyper:
                rep = data.get_candidate_representation(
                    id2text[candidate_id],
                    tokenizer,
                    max_cand_length,
                    overlap_sections,
                    sub_event,
                    candidate_title=id2title[candidate_id],
                    hyperlinks=id2hyper[candidate_id],
                    year=year
                )
            else:
                rep = data.get_candidate_representation(
                    id2text[candidate_id],
                    tokenizer,
                    max_cand_length,
                    overlap_sections,
                    sub_event,
                    candidate_title=id2title[candidate_id],
                    hyperlinks=None,
                    year=year
                )
            tokens_ids = rep["ids"]
            assert len(tokens_ids) == max_cand_length
            candidates.append(tokens_ids)

        label_input_list.append(label_id)
        candidate_input_list.append(candidates)

        idx += 1
        sys.stdout.write("{}/{} \r".format(idx, len(labels)))
        sys.stdout.flush()

    label_input_list = np.asarray(label_input_list)
    candidate_input_list = np.asarray(candidate_input_list)

    return label_input_list, candidate_input_list


def filter_crossencoder_tensor_input(
    context_input_list, label_input_list, candidate_input_list
):
    # remove the - 1 : examples for which gold is not among the candidates
    context_input_list_filtered = [
        x
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    label_input_list_filtered = [
        z
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    candidate_input_list_filtered = [
        y
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    return (
        context_input_list_filtered,
        label_input_list_filtered,
        candidate_input_list_filtered,
    )


def prepare_crossencoder_data(
    tokenizer, samples, labels, nns, id2title, id2text, id2hyper, keep_all=False, args=None
):

    # encode mentions
    context_input_list = prepare_crossencoder_mentions(tokenizer, samples, max_context_length=args["max_context_length"])

    # encode candidates (output of biencoder)
    label_input_list, candidate_input_list = prepare_crossencoder_candidates(
        tokenizer, labels, nns, id2title, id2text, id2hyper, max_cand_length=args["max_cand_length"]
    )

    if not keep_all:
        # remove examples where the gold entity is not among the candidates
        (
            context_input_list,
            label_input_list,
            candidate_input_list,
        ) = filter_crossencoder_tensor_input(
            context_input_list, label_input_list, candidate_input_list
        )
    else:
        label_input_list = [0] * len(label_input_list)
    context_input = torch.LongTensor(context_input_list)
    label_input = torch.LongTensor(label_input_list)
    candidate_input = torch.LongTensor(candidate_input_list)

    return (
        context_input,
        candidate_input,
        label_input,
    )
