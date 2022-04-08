import os

folder_path = 'txt_data'


def get_files():
    with open(folder_path + '/inf.txt', encoding='utf-8', mode='r') as f:
        names = str(f.read())
        print(names)
        name_list = [folder_path + os.sep + name + '.txt' for name in names.split(',')]
        return name_list


if __name__ == '__main__':
    get_files()
