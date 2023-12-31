import os
import argparse
import json
import re
import copy

from transformers import pipeline

model_checkpoint = "./model" # 체크포인트명 비공개
ner = pipeline("ner", model=model_checkpoint, aggregation_strategy="simple")
#ner.save_pretrained('./model')

def ner_tagger(sent:str):
    global ner
    ner_tags = ner(sent)

    for dict_ in ner_tags:
        dict_.pop('score')

    sent2 = list(sent)
    for dict_ in ner_tags[::-1]: # 인덱스 안깨지게 뒤에서부터
        sent2.insert(dict_['end'], "</span>")
        sent2.insert(
            dict_['start'], 
            '<span type="' + dict_['entity_group'] + '">'
        )
    return ner_tags #''.join(sent2)



CRIME_VOCAB_PATH = "data/crime_vocab.txt"
kw_list = []

def load_list():
    global CRIME_VOCAB_PATH, kw_list
    fin = open(CRIME_VOCAB_PATH, "r", encoding='utf-8')
    lines = fin.readlines()

    kw_list = []
    for line in lines:
        word, ner_tag, _ = line.split('\t')
        dict_ = {
            "entity_group": ner_tag,
            "word": word,
        }
        kw_list.append(dict_)

    return

def kw_tagger(sent:str) -> str:
    if not kw_list:   # 최초에 kw_list load
        load_list()

    kw_tags = []    # Keyword Tag List
    for dict_ in kw_list:
        mat_obj = re.finditer(pattern=dict_["word"], string=sent)
        for match in mat_obj:
            # print(match.start(), match.end(), match.span(), match.group(), match.groups())
            dict_["start"] = match.start()
            dict_["end"] = match.end()
            kw_tags.append(copy.deepcopy(dict_)) # deepcopy하여 같은 단어 여러번 매치될 때 덮어써지는 것 방지

    return kw_tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='모바일 메신저 서비스의 내보내기 파일 파싱 결과를 전달받아 NER Tag List를 삽입합니다.'
    )
    parser.add_argument('filename', type=str, help='filename to parse')
    args = parser.parse_args()
    print("loaded json from : ", args.filename)

    with open(args.filename, "r", encoding='utf-8') as fin:
        json_obj = json.load(fin)
        messages=[]
        for idx, msg_obj in enumerate(json_obj["messages"]):

            ner_tags = ner_tagger(msg_obj["message"])
            msg_obj["ner_tags"] = ner_tags  # NER Tag List 삽입

            kw_tags = kw_tagger(msg_obj["message"])
            msg_obj["kw_tags"] = kw_tags  # Keyword Tag List 삽입

            messages.append(copy.deepcopy(msg_obj)) # deepcopy하여 같은 모양의 obj가 여러번 나올 때 덮어써지는 것 방지

    res = json_obj
    res["messages"] = messages # deepcopy한 애를 넣어서 중간에 이상하게 수정되는거 방지
    print(res)

    # save results to json file
    fpath = "./result/"
    fname = args.filename.rstrip(".json")+"_tagged"+".json"
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    try:
        with open(os.path.join(fpath, fname), "w", encoding='utf-8') as fout:
            json.dump(res, fout, ensure_ascii=False, indent=1)
    except IOError:
        print("No such file or directory", os.path.join(fpath, fname))

    print("saved results to : ", fname)