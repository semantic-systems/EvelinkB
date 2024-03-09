import json
import os
import pickle
import flair
import torch
import wikipediaapi
import re
import dateutil.parser as dparser
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

wiki_wiki = wikipediaapi.Wikipedia('EvelinkB2 (evelinkb@gmail.com)', 'en')


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


def GetAllSections(sec, level=0):
    section_titles = []
    for s in sec:
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
    if 100 < len(data_sentence) < 15000:
        sentence = Sentence(data_sentence)
        # predict NER tags
        tagger.predict(sentence)
        # print predicted NER spans
        # print('The following events are found:
        # iterate over entities and print
        for entity in sentence.get_spans('ner'):
            if entity.tag == "EVENT" and entity.score > 0.3:
                all_events.append((entity.text, entity.score))
    else:
        chunks, chunk_size = len(data_sentence), len(data_sentence) // 8
        sub_sentences = [data_sentence[i:i + chunk_size] for i in range(0, chunks, chunk_size)]
        for s in sub_sentences:
            sentence = Sentence(s)
            # predict NER tags
            tagger.predict(sentence)
            # print predicted NER spans
            # print('The following events are found:
            # iterate over entities and print
            for entity in sentence.get_spans('ner'):
                if entity.tag == "EVENT" and entity.score > 0.3:
                    all_events.append((entity.text, entity.score))

    all_events_unique = []
    # removing duplicates
    if len(all_events) > 0:
        seen = set()
        # using list comprehension finding unique tuples
        all_events_unique = [(a, b) for a, b in all_events if not (a in seen or seen.add(a))]
    return all_events_unique


def save_data(t2sub, t2sections, t2year):
    try:
        with open('./newdata/subevents/sub_event_p21.json', 'w') as f:
            json.dump(t2sub, f)
    except:
        save_data_path = os.path.join("./newdata/subevents/", "sub_event_p21.txt")
        outfile = open(save_data_path, 'w')
        outfile.write(f'{t2sub}')
        outfile.close()

    try:
        with open('./newdata/subsections/sub_sections_p21.json', 'w') as f:
            json.dump(t2sections, f)
    except:
        save_data_path = os.path.join("./newdata/subsections/", "sub_sections_p21.txt")
        outfile = open(save_data_path, 'w')
        outfile.write(f'{t2sections}')
        outfile.close()

    try:
        with open('./newdata/date/title_year_p21.json', 'w') as f:
            json.dump(t2year, f)
    except:
        save_data_path = os.path.join("./newdata/date/", "title_year_p21.txt")
        outfile = open(save_data_path, 'w')
        outfile.write(f'{t2year}')
        outfile.close()

    print("Saved")
    return


def func1(title_text, id2title):
    a = 4870000
    b = 4940000
    print("From: ", a, " Till: ", b)
    title_keys = list(title_text.keys())[a:b]
    tagger1 = SequenceTagger.load("flair/ner-english-ontonotes-large")
    t2sub = {}
    t2sections = {}
    t2year = {}
    iter_ = tqdm(title_keys)
    index = 0
    for page in iter_:
        wiki_page = page
        if wiki_page[0] == "'" and wiki_page[-1] == "'":
            wiki_page = wiki_page[1:-1]
        if wiki_page.replace(" ", '_') in id2title[1]:
            wiki_page_id = id2title[1][wiki_page.replace(" ", '_')]

            # Sub Events
            try:
                sub_events = GetSubeventMention(title_text[page], tagger1)
            except:
                sub_events = []
                print("No Sub Event Found for Wiki Page: ", wiki_page)
            if len(sub_events) > 0:
                t2sub[wiki_page_id] = sub_events

            # Get Sections
            sections = GetSectionsFromWiki(wiki_page.replace("_", ' '))
            all_sections = []
            if sections is not None:
                all_sections = GetAllSections(sections)

            if len(all_sections) > 0:
                t2sections[wiki_page_id] = all_sections

            # Title Year
            year = GetDate(wiki_page.replace("_", ' '))
            if year is not None:
                t2year[wiki_page_id] = year

        if index % 20000 == 0:
            print("Saving to File with Index: ", index)
            save_data(t2sub, t2sections, t2year)
        index += 1

    save_data(t2sub, t2sections, t2year)
    return t2sub, t2year, t2sections


if __name__ == "__main__":
    text_file = open('data/wikipedia/wiki/title_text.json', 'r')
    title_text = json.load(text_file)
    print('Text loaded.')

    id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
    id2title = pickle.load(id2title_file)
    print('ID to title loaded.')

    subs, years, sections = func1(title_text, id2title)
    print("SubEvents Length: ", len(list(subs.keys())))
    print("Years Length: ", len(list(years.keys())))
    print("Sections Length: ", len(list(sections.keys())))
