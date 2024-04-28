import json, pickle
import io


def ConvertNYTData(outfile_path):
    id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
    id2title = pickle.load(id2title_file)
    print('ID to title loaded.')

    t2sub_events_file = open('data/wikipedia/extra/extra_sub_events.json', 'r')
    t2sub_events = json.load(t2sub_events_file)
    print('Sub Events loaded.')

    t2sub_sections_file = open('data/wikipedia/extra/extra_sub_sections.json', 'r')
    t2sub_sections = json.load(t2sub_sections_file)
    print('Sub Sections loaded.')

    t2date_file = open('data/wikipedia/extra/extra_date.json', 'r')
    t2date = json.load(t2date_file)
    print('Dates loaded.')

    data = []
    with io.open('data/nyt/nyt_test.jsonl', mode="r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    print('Data loaded.')

    for dp in data:

        # if label_id in dp and label_title in dp:
        label_id = dp['label_id']
        if label_id in t2sub_events:
            dp['sub_events'] = t2sub_events[label_id]
            # print(label_id, " For ", dp['sub_events'])
        else:
            dp['sub_events'] = []
        # elif id2title[1][label_title] in t2sub_events:
        #     dp['sub_events'] = t2sub_events[id2title[1][label_title]]
        #     print(label_title, " For ", dp['sub_events'])

        if label_id in t2sub_sections:
            dp['sub_sections'] = t2sub_sections[label_id]
            # print(label_id, " For ", dp['sub_sections'])
        else:
            dp['sub_sections'] = []
        # elif id2title[1][label_title] in t2sub_sections:
        #     dp['sub_sections '] = t2sub_sections[id2title[1][label_title]]
        #     print(label_title, " For ", dp['sub_sections'])

        if label_id in t2date:
            dp['year'] = t2date[label_id]
            # print(label_id, " For ", dp['year'])
        else:
            dp['year'] = None
        # elif id2title[1][label_title] in t2date:
        #     dp['year'] = t2date[id2title[1][label_title]]
        #     print(label_title, " For ", dp['year'])

    with open(outfile_path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')


def GenerateBlinkData(infile_path, outfile_path):
    text_file = open('data/wikipedia/wiki/title_text.json', 'r')
    title_text = json.load(text_file)
    print('Text loaded.')

    id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
    id2title = pickle.load(id2title_file)
    print('ID to title loaded.')

    t2hyperlinks_file = open('data/wikipedia/wiki/t2hyperlinks.json', 'r')
    t2h = json.load(t2hyperlinks_file)
    print('Hyperlinks loaded.')

    t2sub_events_file = open('data/wikipedia/extra/extra_sub_events.json', 'r')
    t2sub_events = json.load(t2sub_events_file)
    print('Sub Events loaded.')

    t2sub_sections_file = open('data/wikipedia/extra/extra_sub_sections.json', 'r')
    t2sub_sections = json.load(t2sub_sections_file)
    print('Sub Sections loaded.')

    t2date_file = open('data/wikipedia/extra/extra_date.json', 'r')
    t2date = json.load(t2date_file)
    print('Dates loaded.')

    infile = open(infile_path, 'r')
    clusters = json.load(infile)

    query_id = 0
    json_list = []

    for event in clusters:
        event_form = event
        if event[0] == "'" and event[-1] == "'":
            event_form = event[1:-1]

        hyperlinks = []
        for idx, title in enumerate(t2h[id2title[1][event_form]]):
            if idx > 10:
                break
            link = t2h[id2title[1][event_form]][title]
            if link['end'] > 2000:
                continue
            hyperlinks.append((title, title_text[event_form.replace("_", ' ')][link['start']: link['end']]))

        sub_events = []
        if id2title[1][event_form] in t2sub_events:
            sub_events = t2sub_events[id2title[1][event_form]]

        sub_sections = []
        if id2title[1][event_form] in t2sub_sections:
            sub_sections = t2sub_sections[id2title[1][event_form]]

        year = None
        if id2title[1][event_form] in t2date:
            year = t2date[id2title[1][event_form]]

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
                "sub_events": sub_events,
                "sub_sections": sub_sections,
                "year": year,
                "label_id": id2title[1][event_form],
                "hyperlinks": hyperlinks,
                "entities": entities
            })
            query_id += 1
    print(query_id)

    with open(outfile_path, 'w') as outfile:
        for entry in json_list:
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    ConvertNYTData('data/nyt/new_nyt.jsonl')
    # GenerateBlinkData('data/wikipedia/preprocessed/wiki_valid.json', 'out/valid.jsonl')
    # print("Valid Completed")
    # GenerateBlinkData('data/wikipedia/preprocessed/wiki_test.json', 'out/test.jsonl')
    # print("Test Completed")
    # GenerateBlinkData('data/wikipedia/preprocessed/wiki_train.json', 'out/train.jsonl')
    # print("Train Completed")

