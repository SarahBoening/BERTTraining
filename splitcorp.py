import os
import random
import glob

def get_paths(lang, path):
    """
    :param lang: programming language aka file ending, e.g. 'java', 'py'
    :param path: starting directory
    :return: paths of all files with the ending .lang
    """
    print('loading paths...')
    paths = [f for f in glob.glob(path + '**/*.' + lang, recursive=True)]
    return paths

def save_files(data_set, output_path, part):

    for file in data_set:
        print("wrinting file " + file)
        name = file.split("\\")[-1]
        #name = file.split("/")[-1]
        file_path = os.path.join(output_path+"\\"+part+"\\", name)
        #file_path = os.path.join(output_path+"/"+part+"/", name)
        output_file = open(file_path, "w", encoding="UTF-8")
        try:
            with open(file, "r", encoding="UTF-8", errors='ignore') as f:
                text = f.read()
            output_file.write(text)
            output_file.close()
        except Exception as e:
            print("Error processing file " + file)

if __name__ == '__main__':
    path_to_java_corpus = "E:\\Hiwi\\BERT_undCo\\java_projects.tar\\java_projects\\java_projects\\"
    output_folder = "E:\\Hiwi\\BERT_undCo\\Split_Java_Corpus\\"
    #path_to_java_corpus = "/home/nilo4793/media/java_projects"
    #output_folder = "/home/nilo4793/media/Split_Corpus"

    file_list = get_paths('java', path_to_java_corpus)
    random.seed(666)
    random.shuffle(file_list)
    thresh = int(len(file_list)/100*80)
    train_set = file_list[:thresh]
    eval_set = file_list[thresh+1:]
    print(len(train_set))
    print(len(eval_set))
    print("saving train split...")
    save_files(train_set, output_folder, "train")
    print("saving eval split...")
    save_files(eval_set, output_folder, "eval")

