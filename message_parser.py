# message_parser.py
import os
import argparse
import datetime as dt
import json
import csv
import re

FPATHS = {
    "KAKAO_WIN": [],
    "KAKAO_MAC": [],
    "KAKAO_AND": [],
    "KAKAO_IOS": [],
    "TELEGRAM_JSON": [],
}


def kakao_win(fin):
    lines = fin.readlines()

    messages = []
    title, saved_time, chat_date = "", "", ""
    name, time, message = "", "", ""
    for idx, line in enumerate(lines):
        line = line.rstrip('\n')
        if line=="":
            continue
        elif idx == 0 and line.endswith("님과 카카오톡 대화"):
            title = line.rstrip("님과 카카오톡 대화").rstrip()
            #print(idx, title)
        elif idx == 1 and line.startswith("저장한 날짜 : "):
            saved_time = line.lstrip("저장한 날짜 : ")
            saved_time = dt.datetime.strptime(saved_time, "%Y-%m-%d %H:%M:%S")
            #print(idx, saved_time)
        elif line.startswith("---------------") and line.endswith("---------------"):
            date = line.strip("---------------").strip()
            try:
                date = dt.datetime.strptime(date[:-4], "%Y년 %m월 %d일")
                #print(date)
            except:
                if all(x != "" for x in [name, time, message]):
                    message += "\n"+line
                    continue
            chat_date = date
        elif line.endswith("님이 들어왔습니다.") or line.endswith("님이 나갔습니다."):
            continue
        elif line.startswith("["):
            sep1 = line[1:line.index("]")]
            temp = line[line.index("]")+2:]
            if not temp.startswith("["):
                print(line)
                continue
            sep2 = temp[1:temp.index("]")]
            sep3 = temp[temp.index("]")+2:]

            if all(x != "" for x in [name, time, message]):
                obj = {
                    'name': name,
                    'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': message
                }
                messages.append(obj)

            try:
                ap, hours, minutes = sep2.split()[0], int(sep2.split()[1].split(':')[0]), int(sep2.split()[1].split(':')[1])
            except ValueError:
                if all(x != "" for x in [line, name, time, message]):
                    message += "\n"+line
                    continue
            if ap == "오후":
                hours += 12
            elif ap != "오전":
                print("time parse error")
                if all(x != "" for x in [line, name, time, message]):
                    message += "\n"+line
                continue
            if chat_date == "":
                print("no chat_date")
                continue
            #print(chat_date)
            time = chat_date + dt.timedelta(minutes=minutes, hours=hours)
            
            name, time, message = sep1, time, sep3
        elif all(x != "" for x in [line, name, time, message]):
            message += "\n"+line
    if all(x != "" for x in [name, time, message]):
        obj = {
            'name': name,
            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'message': message,
        }
        messages.append(obj)

    res = {
        'title': title,
        'saved_time': saved_time.strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages,
    }
    return res

def kakao_mac(fin):
    lines = csv.reader(fin)

    messages = []
    title, saved_time = os.path.basename(fin.name), dt.datetime.now()
    
    for idx, line in enumerate(lines):
        if idx == 0: # column name 라인 제거
            continue

        try:
            sep1, sep2, sep3 = line[0], line[1], line[2]
        except:
            print("csv parse error")
            continue
        name = sep2
        try:
            time = dt.datetime.strptime(sep1, "%Y-%m-%d %H:%M:%S")
        except:
            print("time parse error:", str(idx)+"-th line")
            continue
        message = sep3

        obj = {
            'name': name,
            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'message': message,
        }
        messages.append(obj)

    res = {
        'title': title,
        'saved_time': saved_time.strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages,
    }
    return res

def kakao_and(fin):
    def time_parser_and(time_str):
        try:
            sep1, sep2 = ' '.join(time_str.split()[0:3]), ' '.join(time_str.split()[3:5])
            sep1 = dt.datetime.strptime(sep1, "%Y년 %m월 %d일")
            ap, hours, minutes = sep2.split()[0], int(sep2.split()[1].split(':')[0]), int(sep2.split()[1].split(':')[1])
            if ap == "오후":
                hours += 12
            elif ap != "오전":
                return False
            saved_time = sep1 + dt.timedelta(minutes=minutes, hours=hours)
        except:
            return False
        return saved_time
    
    lines = fin.readlines()

    messages = []
    title, saved_time, chat_date = os.path.basename(fin.name), dt.datetime.now(), ""
    name, time, message = "", "", ""
    for idx, line in enumerate(lines):
        line = line.rstrip('\n')
        if line=="":
            continue
        elif idx == 0 and line.endswith("님과 카카오톡 대화"):
            title = line.rstrip("님과 카카오톡 대화").rstrip()
            #print(idx, title)
        elif idx == 1 and line.startswith("저장한 날짜 : "):
            time_str = line.lstrip("저장한 날짜 : ")
            res_time = time_parser_and(time_str)
            if res_time == False:
                print("time parse error in: ", idx)
                continue
            else:
                saved_time = res_time
                #print(idx, saved_time)
        elif line.endswith("님이 들어왔습니다.") or line.endswith("님이 나갔습니다."):
            continue
        elif ', ' in line and ' : ' in line:
            sep1, sep2, sep3 = "", "", ""
            try:
                sep1, temp = line.split(', ', 1)
                sep2, sep3 = temp.split(' : ', 1)
            except:
                if all(x != "" for x in [line, name, time, message]):
                    message += "\n"+line
                continue

            res_time = time_parser_and(sep1)
            if res_time == False:
                print("time parse error in: ", idx)
                if all(x != "" for x in [line, name, time, message]):
                    message += "\n"+line
                continue

            if all(x != "" for x in [name, time, message]):
                obj = {
                    'name': name,
                    'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': message
                }
                messages.append(obj)
            name, time, message = sep2, res_time, sep3
        
        elif all(x != "" for x in [line, name, time, message]):
            if time_parser_and(line) != False: # 시간 표시하는 라인이면 pass
                continue
            message += "\n"+line
    if all(x != "" for x in [name, time, message]):
        obj = {
            'name': name,
            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'message': message,
        }
        messages.append(obj)

    res = {
        'title': title,
        'saved_time': saved_time.strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages,
    }
    return res

def kakao_ios(fin):
    def time_parser_ios(time_str):
        try:
            sep1, sep2 = ' '.join(time_str.split()[0:3]), ' '.join(time_str.split()[3:5])
            sep1 = dt.datetime.strptime(sep1, "%Y. %m. %d.")
            ap, hours, minutes = sep2.split()[0], int(sep2.split()[1].split(':')[0]), int(sep2.split()[1].split(':')[1])
            if ap == "오후":
                hours += 12
            elif ap != "오전":
                return False
            saved_time = sep1 + dt.timedelta(minutes=minutes, hours=hours)
        except:
            return False
        return saved_time
    
    def date_checker_ios(date_str):
        if bool(re.match('([0-9]+년)\s([0-9]+월)\s([0-9]+일)\s([월화수목금토일]요일)', date_str)):
            print("[matches]", date_str)
            return True
        return False
    
    lines = fin.readlines()

    messages = []
    title, saved_time, chat_date = os.path.basename(fin.name), dt.datetime.now(), ""
    name, time, message = "", "", ""
    for idx, line in enumerate(lines):
        line = line.rstrip('\n')
        if line=="":
            continue
        elif idx == 0 and line.endswith("님과 카카오톡 대화"):
            title = line.rstrip("님과 카카오톡 대화").rstrip()
            #print(idx, title)
        elif idx == 1 and line.startswith("저장한 날짜 : "):
            time_str = line.lstrip("저장한 날짜 : ")
            res_time = time_parser_ios(time_str)
            if res_time == False:
                print("time parse error in: ", idx)
                continue
            else:
                saved_time = res_time
                #print(idx, saved_time)
        elif line.endswith("님이 들어왔습니다.") or line.endswith("님이 나갔습니다."):
            continue
        elif ', ' in line and ' : ' in line:
            sep1, sep2, sep3 = "", "", ""
            try:
                sep1, temp = line.split(', ', 1)
                sep2, sep3 = temp.split(' : ', 1)
            except:
                if all(x != "" for x in [line, name, time, message]):
                    message += "\n"+line
                continue

            res_time = time_parser_ios(sep1)
            if res_time == False:
                print("time parse error in: ", idx)
                if all(x != "" for x in [line, name, time, message]):
                    message += "\n"+line
                continue

            if all(x != "" for x in [name, time, message]):
                obj = {
                    'name': name,
                    'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': message
                }
                messages.append(obj)
            name, time, message = sep2, res_time, sep3
        
        elif all(x != "" for x in [line, name, time, message]):
            if date_checker_ios(line): # 날짜 표시하는 라인이면 pass
                continue
            message += "\n"+line
    if all(x != "" for x in [name, time, message]):
        obj = {
            'name': name,
            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'message': message,
        }
        messages.append(obj)

    res = {
        'title': title,
        'saved_time': saved_time.strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages,
    }
    return res

def telegram_json(fin):
    json_obj = json.load(fin)
    #print(lines)

    title = json_obj["name"]
    saved_time = dt.datetime.now()
    messages = []
    for idx, msg_obj in enumerate(json_obj["messages"]):
        if any(key not in msg_obj for key in ["from", "date", "text"]):
            print("> key not exist, so pass this msg : ", msg_obj)
            continue
        name = msg_obj["from"]
        time = dt.datetime.strptime(msg_obj["date"], "%Y-%m-%dT%H:%M:%S")
        text_entities = msg_obj["text_entities"]
        for text_entity in text_entities:
            message = text_entity["text"]
            if message.strip(" ") == "":
                #print("> passing empty text : ", message)
                continue
            message = ' '.join(message.split())
            obj = {
                'name': name,
                'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'message': message,
            }
            messages.append(obj)

    res = {
        'title': title,
        'saved_time': saved_time.strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages,
    }
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='모바일 메신저 서비스의 내보내기 파일을 파싱합니다.'
    )
    parser.add_argument('-p', '--platform', type=str, default='ios', help='select platform in ["win", "mac", "and", "ios", "telegram"]')
    parser.add_argument('filename', type=str, help='filename to parse')
    args = parser.parse_args()
    print(args.platform, args.filename)

    res = {}
    try:
        with open(args.filename, "r", encoding='utf-8') as fin:
            if args.platform == 'win':
                res = kakao_win(fin)

            elif args.platform == 'mac':
                res = kakao_mac(fin)

            elif args.platform == 'and':
                res = kakao_and(fin)

            elif args.platform == 'ios':
                res = kakao_ios(fin)

            elif args.platform == 'telegram':
                res = telegram_json(fin)

            else :
                print('wrong platform\nselect platform in ["win", "mac", "and", "ios", "telegram"]')
                exit()
                
    except IOError:
        print("No such file :", args.filename)
    
    # save results to json file
    fpath = "./result/"
    fname = res["title"]+"_"+res["saved_time"].split()[0]+".json"
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    try:
        with open(os.path.join(fpath, fname), "w", encoding='utf-8') as fout:
            json.dump(res, fout, ensure_ascii=False, indent=1)
    except IOError:
        print("No such file or directory", os.path.join(fpath, fname))