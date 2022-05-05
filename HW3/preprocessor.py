import os

import chardet

folder_path = 'txt_data'
stopwords_files = ['baidu_stopwords.txt']
stopwords_list = ["的", "了", "在", "是", "我", "有", "和", "就",
                  "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这",
                  "罢", "这", '在', '又', '在', '得', '那', '他', '她', '不', '而', '道', '与', '之', '⻅', '却', '问', '可', '但', '没', '啦', '给', '来', '既',
                  '叫', '只', '中', '么', '便', '听', '为', '跟', '个', '甚', '下', '还', '过', '向', '如此', '已', '位', '对', '如何', '将', '岂', '哪', '似', '以免', '均', '虽然', '即',
                  '由', '再', '使', '从', '麽', '其实', '阿', '被']


def get_files():
    with open(folder_path + '/inf.txt', encoding='utf-8', mode='r') as f:
        names = str(f.read())
        print(names)
        name_list = [folder_path + os.sep + name + '.txt' for name in names.split(',')]
        return name_list


if __name__ == '__main__':
    get_files()


def import_stopwords():
    for sf in stopwords_files:
        with open(sf, 'r') as f:
            stopwords_list.extend([word.strip('\n') for word in f.readlines()])
    print(stopwords_list)


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def get_texts():
    import_stopwords()
    corpus_context_dict = {}
    id_corpus_dict = {}
    id = 0
    for file in get_files():
        simple_name = str(file).split(os.sep)[1].split('.')[0]
        with open(file, 'rb') as f:
            context = f.read()
            real_encode = chardet.detect(context)['encoding']
            context = context.decode(real_encode, errors='ignore')
            new_context = ''
            for c in context:
                if is_chinese(c):
                    new_context += c
            # for sw in stopwords_list:
            #     new_context = new_context.replace(sw, '')
            corpus_context_dict[simple_name] = new_context
            id_corpus_dict[id] = simple_name
            id += 1
        print(id)
    return corpus_context_dict, id_corpus_dict


if __name__ == '__main__':
    import_stopwords()
    get_texts()
