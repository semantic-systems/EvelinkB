import json
import io, os
import torch
import requests
import dateutil.parser as dparser


# from thefuzz import fuzz
# import datefinder
# from levenshtein import levenshtein


def callNER(text):
    NER_HTTP = "http://dickens.seas.upenn.edu:4033/ner/"

    input = {"lang": "eng", "model": "onto_ner",
             "text": text.replace(' [ ', ' ').replace(' ] ', ' ')}
    res_out = requests.post(NER_HTTP, json=input)

    try:
        res_json = res_out.json()
        events = []
        for i in range(len(res_json['views'])):
            if res_json['views'][i]['viewName'] == 'NER_CONLL':
                for element in res_json['views'][i]['viewData'][0]['constituents']:
                    ner_type = element['label']
                    surface_form = ' '.join(res_json['tokens'][element['start']: element['end']])
                    events.append({"type": ner_type, "form": surface_form})
        return events
    except:
        return []


if __name__ == "__main__":
    # print(dparser.parse("World War 1", fuzzy=True))
    # print(dparser.parse("66th Annual Grammy Awards", fuzzy=True).year < 2024)
    # print(dparser.parse("2020 Olympic Games", fuzzy=True).year < 2024)
    # print(dparser.parse("1928 Summer Olympics", fuzzy=True).year)
    # print(fuzz.ratio("Summer Olympics", "2020 Summer Olympics"))
    # path_to_json = './newdata/subsections/'
    # json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    # print(json_files)  # for me this prints ['foo.json']
    # # Create an empty list to store the Python objects.
    # python_objects = []
    #
    json_files = ['sub_sections_combined.json', 'sub_sections_combined_base.json']
    # Load each JSON file into a Python object.
    big_json = {}
    for json_file in json_files:
        print(json_file)
        with open("./newdata/" + json_file, "r") as f:
            onejson = json.load(f)
            print(len(onejson.keys()))
            for i in onejson:
                if i not in big_json:
                    big_json[i] = onejson[i]
                else:
                    print("ID Exist ", i, " In Json File ", json_file, " with value ", onejson[i])

    print(len(big_json.keys()))
    # # Dump all the Python objects into a single JSON file.
    with open("./newdata/all_sub_sections.json", "w") as f:
        json.dump(big_json, f)

    # data_to_link = []
    # with io.open('./sub_event_p1.json', mode="r", encoding="utf-8") as file:
    #     for line in file:
    #         data_to_link.append(json.loads(line.strip()))
    #
    # print(data_to_link[1])
    #
    # subevents_file = open('./newdata/sub_events_combined.json', 'r')
    # subevents = json.load(subevents_file)
    # print('Combined Text loaded.', len(subevents.keys()))
    # print("3019583:", subevents['3019583'])
    # print("188278:", subevents['188278'])
    # print("27472362:", subevents['27472362'])

    # print("Length data_to_link", len(data_to_link))
    #
    # infile = open('./out/valid_result_top30.json', 'r')
    # clusters = json.load(infile)
    #
    # print("Loaded Data ,Converting  Now!!!")
    #
    # idx = 0
    # for t_idx, title in enumerate(clusters):
    #     for mention in clusters[title]:
    #         data_to_link[idx]['edl'] = mention['edl']
    #         data_to_link[idx]['scores'] = mention['scores']
    #         idx += 1
    #
    # print("Length of Mentions in CLuster: ", idx)
    #
    # with open('./out/valid_result_dp_top30.json', 'w') as outfile:
    #     for entry in data_to_link:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    #
    # print("Saved File!!")

    # fname = os.path.join("./out/old/base/", "train.t7")
    # train_data = torch.load(fname)
    # context_input = train_data["context_vecs"]
    # candidate_input = train_data["candidate_vecs"]
    # label_input = train_data["labels"]
    #
    # print("Context Input Size", context_input.size())
    # print("Candidate Input Size", candidate_input.size())
    # print("Label Input Size", label_input.size())
    # print("Label Input", label_input)

    # fname = os.path.join("./out/old/base/", "new_train.t7")
    # train_data = torch.load(fname)
    # context_input = train_data["context_vecs"]
    # candidate_input = train_data["candidate_vecs"]
    # label_input = train_data["labels"]
    #
    # print("Context Input Size", context_input.size())
    # print("Candidate Input Size", candidate_input.size())
    # print("Label Input Size", label_input.size())
    # print("Label Input", label_input)

    # fname1 = os.path.join("./out/exp5_date_and_sections/train/", "new_train_p1_new.t7")
    # train_data1 = torch.load(fname1)
    # context_input1 = train_data1["context_vecs"]
    # candidate_input1 = train_data1["candidate_vecs"]
    # label_input1 = train_data1["labels"]
    #
    # print("Context Input Size", context_input1.size())
    # print("Candidate Input Size", candidate_input1.size())
    # print("Label Input Size", label_input1.size())
    # print("Label Input", label_input1)
    #
    # fname2 = os.path.join("./out/exp5_date_and_sections/train/", "new_train_p2_new.t7")
    # train_data2 = torch.load(fname2)
    # context_input2 = train_data2["context_vecs"]
    # candidate_input2 = train_data2["candidate_vecs"]
    # label_input2 = train_data2["labels"]

    # print("Context Input Size", context_input2.size())
    # print("Candidate Input Size", candidate_input2.size())
    # print("Label Input Size", label_input2.size())
    # print("Label Input", label_input2)
    #
    # fname3 = os.path.join("./out/exp5_date_and_sections/train/", "new_train_p3_new.t7")
    # train_data3 = torch.load(fname3)
    # context_input3 = train_data3["context_vecs"]
    # candidate_input3 = train_data3["candidate_vecs"]
    # label_input3 = train_data3["labels"]
    #
    # print("Context Input Size", context_input3.size())
    # print("Candidate Input Size", candidate_input3.size())
    # print("Label Input Size", label_input3.size())
    # print("Label Input", label_input3)
    #
    # fname4 = os.path.join("./out/exp5_date_and_sections/train/", "new_train_p4_new.t7")
    # train_data4 = torch.load(fname4)
    # context_input4 = train_data4["context_vecs"]
    # candidate_input4 = train_data4["candidate_vecs"]
    # label_input4 = train_data4["labels"]

    # print("Context Input Size", context_input4.size())
    # print("Candidate Input Size", candidate_input4.size())
    # print("Label Input Size", label_input4.size())
    # print("Label Input", label_input4)
    #
    # new_context = torch.cat((context_input1, context_input2, context_input3, context_input4), 0)
    # new_candidate = torch.cat((candidate_input1, candidate_input2, candidate_input3, candidate_input4), 0)
    # new_label = torch.cat((label_input1, label_input2, label_input3, label_input4), 0)
    #
    # print("New Context Input Size", new_context.size())
    # print("New Candidate Input Size", new_candidate.size())
    # print("New Label Input Size", new_label.size())
    # print("New Label Input", new_label)

    # nn_data = {
    #     'context_vecs': new_context,
    #     'candidate_vecs': new_candidate,
    #     'labels': new_label
    # }
    #
    # try:
    #     save_data_path = os.path.join("./out/exp5_date_and_sections/", "new_train.t7")
    #     torch.save(nn_data, save_data_path)
    # except:
    #     save_data_path = os.path.join("./out/exp5_date_and_sections/", "new_train.json")
    #     outfile = open(save_data_path, 'w')
    #     json.dump(nn_data, outfile)

    # text_file = open('data/wikipedia/wiki/title_text.json', 'r')
    # title_text = json.load(text_file)
    # print(len(title_text))
    #
    # events = callNER(title_text["World War II"])
    # print(events)

    # infile = open("./out/train_result_dp_top30.json", 'r')
    # data = json.load(infile)
    # idx = 0
    # for t_idx, dp in enumerate(data):
    #     print(dp)
    #     break
    # print("Length Predictions", idx)
