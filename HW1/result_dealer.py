import csv
import json

if __name__ == '__main__':
    real_dict = {}
    with open('result.txt') as f:
        ret_dic = dict(json.load(f))
        print(ret_dic)
        for model_type in ret_dic.keys():
            for file_name in dict(ret_dic.get(model_type)).keys():
                if not real_dict.keys().__contains__(file_name):
                    real_dict[file_name] = []
                real_dict[file_name].append(dict(ret_dic.get(model_type)).get(file_name))
        print(real_dict)

        d = []
        for r in real_dict.keys():
            dd = [r]
            for rr in real_dict.get(r):
                dd.append(rr)
            d.append(dd)

        print(d)

        with open('result.csv','w',encoding='utf-8') as f2:
            writer = csv.writer(f2)
            writer.writerow(['uni','bi','tri'])
            writer.writerows(d)

