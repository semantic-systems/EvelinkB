import sys, io, os
import json
import pickle
import nltk
import wikipediaapi
from flair.data import Sentence
from flair.models import SequenceTagger
from cosine import get_cosine_result

wiki_wiki = wikipediaapi.Wikipedia('EvelinkB (evelinkb@gmail.com)', 'en')
def get_hyperlink_with_type(hyperlink, ner_tagger):
    entity_tag = None
    try:
        wp = wiki_wiki.page(hyperlink[0].replace("_",' ').replace("'",'').strip('\"'))
        # wp = wikipedia.WikipediaPage(hyperlink[0].replace("_",' ').replace("'",'').strip('\"'))
    # except wikipedia.exceptions.DisambiguationError:
    except:
        print("No Wiki Page found for hyperlink " + hyperlink)
    else:
        if wp is not None:
            # print("summary",wp.summary)
            sentence = Sentence(wp.summary)
            ner_tagger.predict(sentence)

            if sentence.get_spans('ner') is None:
                print("No NER tag found for hyperlink " + hyperlink)
            # iterate over entities and print
            for entity in sentence.get_spans('ner'):
                if get_cosine_result(entity.text, hyperlink[0].replace("_",' ')) > 0.25 and entity.score > 0.5:
                    print(entity.text, hyperlink[0].replace("_", ' '))
                    print("Cosine Result", get_cosine_result(entity.text, hyperlink[0].replace("_", ' ')))
                    entity_tag = entity.tag
                    break


    new_hyperlink = [hyperlink[0],hyperlink[1],entity_tag]
    return new_hyperlink

if __name__ == '__main__':
    # Load Sequence Tagger
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-large')
    print('Sequence Tagger loaded.')

    with io.open('out/valid.jsonl', mode="r", encoding="utf-8") as file:
        for line in file:
            dp = json.loads(line.strip())
            # print(dp["hyperlinks"])
            for hyperlink in dp["hyperlinks"]:
                hyperlink_with_type = get_hyperlink_with_type(hyperlink,tagger)
                print(hyperlink_with_type)

