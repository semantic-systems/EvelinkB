import json, pickle
from flair.data import Sentence
from flair.models import SequenceTagger
from cosine import get_cosine_result
import wikipediaapi
import time
import traceback
import logging
import dateutil.parser as dparser
import re

wiki_wiki = wikipediaapi.Wikipedia('EvelinkB (evelinkb@gmail.com)', 'en')


def get_hyperlink_with_type(hyperlink, ner_tagger):
    entity_tag = None
    wp = wiki_wiki.page(hyperlink.replace("_", ' ').replace("'", '').strip('\"'))
    if wp is not None:
        if wp.summary != "" or wp.summary is not None:
            # print("summary",wp.summary)
            sentence = Sentence(wp.summary)
            ner_tagger.predict(sentence)

            # iterate over entities and print
            for entity in sentence.get_spans('ner'):
                if get_cosine_result(entity.text, hyperlink.replace("_", ' ')) > 0.3 and entity.score > 0.5:
                    # print(entity.text," ------ ", hyperlink.replace("_", ' '))
                    # print("Cosine Result", get_cosine_result(entity.text, hyperlink.replace("_", ' ')))
                    entity_tag = entity.tag
                    break
        else:
            print("No Summary found for hyperlink " + hyperlink)
            if wp.content != "" or wp.content is not None:
                # print("summary",wp.summary)
                sentence = Sentence(wp.content[:2000])
                ner_tagger.predict(sentence)

                # iterate over entities and print
                for entity in sentence.get_spans('ner'):
                    if get_cosine_result(entity.text, hyperlink.replace("_", ' ')) > 0.3 and entity.score > 0.5:
                        # print(entity.text," ------ ", hyperlink.replace("_", ' '))
                        # print("Cosine Result", get_cosine_result(entity.text, hyperlink.replace("_", ' ')))
                        entity_tag = entity.tag
                        break

    return entity_tag


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


def GetDate(title: str):
    if re.search('\d{2}th', title):
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
    wp = wiki_wiki.page(title)
    sections = wp.sections
    return sections


def GenerateBlinkData(infile_path, outfile_path):
    text_file = open('data/wikipedia/wiki/title_text.json', 'r')
    title_text = json.load(text_file)
    print('Text loaded.')

    id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
    id2title = pickle.load(id2title_file)
    print('ID to title loaded.')

    # t2hyperlinks_file = open('data/wikipedia/wiki/t2hyperlinks.json', 'r')
    # t2h = json.load(t2hyperlinks_file)
    # print('HyperLinks loaded.')

    infile = open(infile_path, 'r')
    clusters = json.load(infile)

    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    print('Sequence Tagger loaded.')

    query_id = 0
    json_list = []
    # sleep_id = 0
    all_years = []
    h = 0
    for event in clusters:
        event_form = event
        if event[0] == "'" and event[-1] == "'":
            event_form = event[1:-1]

        # print("TITLE: " + event_form)
        hyperlinks = []
        year = GetDate(event_form.replace("_", ' ').replace("'", '').strip('\"'))
        if year is not None:
            # print("Date Found: ", year, " From Title: ", event_form)
            all_years.append(year)

        sections = GetSectionsFromWiki(event_form)
        all_sections = []
        if sections is not None:
            all_sections = GetAllSections(sections)
        # print("Sections found: ", len(all_sections))
        # print("Sections", all_sections)
        # h = h + 1
        # if h == 10:
        #     break
        # hyperlink_type = None
        # for idx, title in enumerate(t2h[id2title[1][event_form]]):
        #     if idx > 10:
        #         break
        #     link = t2h[id2title[1][event_form]][title]
        #     # print("Link", link)
        #     if link['end'] > 2000:
        #         continue
        #     # print ("Hyperlink Title",title,"HyperLink appended", title_text[event_form.replace("_", ' ')][link['start']: link['end']])
        #     try:
        #         hyperlink_type = get_hyperlink_with_type(title,tagger)
        #     except Exception as e:
        #         print('No hyperlink Type found for title: ', title)
        #         time.sleep(3)
        #     hyperlinks.append([title, title_text[event_form.replace("_", ' ')][link['start']: link['end']], hyperlink_type])

        sub_event = []
        # # print("Hyperlinks", hyperlinks)
        sub_event = GetSubeventMention(title_text[event_form.replace("_", ' ')], tagger)

        overlap_sections = []
        for item in sub_event:
            if item in all_sections:
                overlap_sections.append(item)

        if overlap_sections:
            print("Overlap Sections: ", overlap_sections, "title: ", event_form)

        for mention in clusters[event]:
            page = title_text[mention['page']]
            entities = []
            for idx, entity in enumerate(mention['entities']):
                form = title_text[mention['page']][entity['start']:entity['end']].lower()
                entity['form'] = form
                entities.append(entity)

            json_list.append({
                "context_left": page[0: mention['start']],
                "mention": page[mention['start']: mention['end']],
                "context_right": page[mention['end']:],
                "label": title_text[event_form.replace("_", ' ')][0:2000],
                "label_title": event_form,
                "sub_events": sub_event,
                "label_id": id2title[1][event_form],
                "hyperlinks": hyperlinks,
                "entities": entities
            })
            query_id += 1
        # sleep_id+=1

        # if sleep_id % 1000 == 0:
        #     print("Sleeping for 10 seconds")
        #     time.sleep(10)

    print(query_id)
    print("Len Years: ", len(all_years))

    # with open(outfile_path, 'w') as outfile:
    #     for entry in json_list:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')


if __name__ == '__main__':
    # GenerateBlinkData('data/wikipedia/preprocessed/wiki_valid.json', 'out/hyp_valid.jsonl')
    GenerateBlinkData('data/wikipedia/preprocessed/wiki_test.json', 'out/hyp_test.jsonl')
    # GenerateBlinkData('data/wikipedia/preprocessed/wiki_train.json', 'out/hyp_train.jsonl')
