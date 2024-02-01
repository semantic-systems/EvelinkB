import json
import io,os
import torch
import requests


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
    # data_to_link=[]
    # with io.open('./out/train_result_dp_top30.json', mode="r", encoding="utf-8") as file:
    #     for line in file:
    #         data_to_link.append(json.loads(line.strip()))
    #
    # print("Length data_to_link",len(data_to_link))
    # print(data_to_link[45000]['edl'])

    # fname = os.path.join("./out/", "train.t7")
    # train_data = torch.load(fname)
    # context_input = train_data["context_vecs"]
    # candidate_input = train_data["candidate_vecs"]
    # label_input = train_data["labels"]
    #
    # print("Context Input Size", context_input.size())
    # print("Candidate Input Size", candidate_input.size())
    # print("Label Input Size", label_input.size())
    # print("Label Input", label_input)

    text_file = open('data/wikipedia/wiki/title_text.json', 'r')
    title_text = json.load(text_file)
    print(len(title_text))

    events = callNER(title_text["World War II"])
    print(events)

    # infile = open("./out/train_result_dp_top30.json", 'r')
    # data = json.load(infile)
    # idx = 0
    # for t_idx, dp in enumerate(data):
    #     print(dp)
    #     break
    # print("Length Predictions", idx)
