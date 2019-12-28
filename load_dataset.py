import os
import glob
import random
import math
# import magic
import pickle as pkl


def process_func(complete_list, index_array, output_folder):
    print("started process... ")
    for file_index, start, end in index_array:
        file_name = "%04d.java_github_1k.raw" % file_index
        fqn = os.path.join(output_folder, file_name)
        print(fqn)
        output_file = open(fqn, "w")
        error = 0
        failed_files = []
        for i in range(start, end):
            try:
                with open(complete_list[i], "r") as f:
                    text = f.read()
                    str = '[CLS] ' + text + '[SEP]'
                    # text = f.read().replace('\n', ' ')
                output_file.write("%s\n" % str)
                # output_file.write("<|endoftext|>\n")
            except Exception as e:
                error += 1
                failed_files.append([complete_list[i], type(e)])
        output_file.close()


def get_paths(lang, path):
    """
    :param lang: programming language aka file ending, e.g. 'java', 'py'
    :param path: starting directory
    :return: paths of all files with the ending .lang
    """
    print('loading paths...')
    paths = [f for f in glob.glob(path + '**/*.' + lang, recursive=True)]
    return paths


def load_dataset(lang, path):
    """
    load files from a directory and store them in a list
    :param lang: programming language aka file ending, e.g. 'java', 'py'
    :param path: starting directory
    :return: list of contests of the files
    """
    print('loading files..')
    paths = get_paths(lang, path)
    data = []
    for path in paths:
        # print(path)
        # enc = get_encoding(path)
        f = open(path, 'r')
        try:
            # data.append(f.read().replace('\n', ' '))
            data.append(f.read())
        except Exception as e:
            print('Error opening file ' + path)
            print(e)
        f.close()
    return data


def get_encoding(path):
    """
    get the encoding of a file
    :param path: path to file
    :return encoding: encoding of the file
    """
    #blob = open(path).read()
    #m = magic.Magic(mime_encoding=True)
    #encoding = m.from_buffer(blob)
    # m = magic.open(magic.MAGIC_MIME_ENCODING)
    # m.load()
    # encoding = m.buffer(blob)
    #print(encoding)
    #return encoding
    pass


if __name__ == '__main__':
    # change
    # input_dir = "G:\\Hiwi\\BERT\\Java_split\\"
    input_dir = "/home/nilo4793/Documents/Bert_Hiwi/corpora/java_projects/"
    output_dir = "/home/nilo4793/Documents/Bert_Hiwi/corpora/Java_split/"

    file_list = []
    # files = load_dataset('py', 'E:\\PyCharm Projects\\')
    # files = load_dataset('java', 'E:\\Hiwi\\BERT_undCo\\java_examples\\')
    # files = load_dataset('java', "G:\\Hiwi\\BERT\\Java_split\\")

    # can be commented out

    print('exporting file list ...')
    
    file_list = get_paths('java', "G:\\Hiwi\\BERT\\Java_split\\")

    with open(output_dir+'file_list.txt', 'w') as f:
        for file in file_list:
            f.write("%s\n" % file)
            
    # can be commented out
    # export the filelist to overcome the re-crawl
    with open("file_list.pkl", "wb") as f:
        pkl.dump(file_list, f)

    with open('file_list.pkl', 'rb') as f:
        file_list = pkl.load(f)

    samples_per_file = int(1000)
    if len(file_list) < samples_per_file:
        samples_per_file = len(file_list)
    print(len(file_list))
    random.shuffle(file_list)

    # can be commented out
    # generate a small random subset
    '''
    print('generating subset... ')
    with open(output_dir+'subset_java_github_10.raw', 'w') as f:
        for i in range(0, 10):
            with open(file_list[i], 'r') as file:
                text = file.read()
                str = '[CLS] ' + text + '[SEP]'
                f.write("%s\n" % str)
    
    with open(output_dir+'subset_no_linebreak_java_github_10.raw', 'w') as fi:
        for i in range(0, 10):
            with open(file_list[i], 'r') as file:
                text = file.read().replace('\n', ' ')
                str = '[CLS] ' + text + ' [SEP]'
                fi.write("%s\n" % str)

    with open(output_dir+'test_subset_java_github_10.raw', 'w') as f:
        for i in range(11, 21):
            with open(file_list[i], 'r') as file:
                text = file.read()
                str = '[CLS] ' + text + '[SEP]'
                f.write("%s\n" % str)

    with open(output_dir+'test_subset_no_linebreak_java_github_10.raw', 'w') as fi:
        for i in range(11, 21):
            with open(file_list[i], 'r') as file:
                text = file.read().replace('\n', ' ')
                str = '[CLS] ' + text + ' [SEP]'
                fi.write("%s\n" % str)

    with open(output_dir+'subset_java_github_30.raw', 'w') as f:
        for i in range(0, 30):
            with open(file_list[i], 'r') as file:
                text = file.read()
                str = '[CLS] ' + text + '[SEP]'
                f.write("%s\n" % str)

    with open(output_dir+'subset_no_linebreak_java_github_30.raw', 'w') as fi:
        for i in range(0, 30):
            with open(file_list[i], 'r') as file:
                text = file.read().replace('\n', ' ')
                str = '[CLS] ' + text + ' [SEP]'
                fi.write("%s\n" % str)

    with open(output_dir+'test_subset_java_github_30.raw', 'w') as f:
        for i in range(31, 61):
            with open(file_list[i], 'r') as file:
                text = file.read()
                str = '[CLS] ' + text + '[SEP]'
                f.write("%s\n" % str)

    with open(output_dir+'test_subset_no_linebreak_java_github_30.raw', 'w') as fi:
        for i in range(31, 61):
            with open(file_list[i], 'r') as file:
                text = file.read().replace('\n', ' ')
                str = '[CLS] ' + text + ' [SEP]'
                fi.write("%s\n" % str)
    '''
    processes = 1
    complete_length = len(file_list)
    print("amount of files in corpus = %d " % complete_length)
    loops = int(complete_length / samples_per_file)
    if loops == 0:
        loops = 1
    print("amount of loops = %d" % loops)
    print("num of processes = %d" % processes)

    loops_per_process = loops / processes
    print("loops per process = %f" % loops_per_process)

    over_sampled = round(loops_per_process % int(loops_per_process) * processes)
    # over_sampled = round(math.remainder(loops_per_process, int(loops_per_process)) * processes)

    distribution = [math.floor(loops_per_process)] * processes
    for i in range(0, over_sampled):
        distribution[i] += 1

    indices = []
    cnt = 0
    for i, e in enumerate(distribution):
        prev_sum = sum(distribution[:i])
        offset = prev_sum * samples_per_file
        process_numbers = []
        for j in range(e):
            start = offset + j * samples_per_file
            process_numbers.append([cnt, start, start + samples_per_file])
            cnt += 1
        indices.append(process_numbers)
    for i in indices:
        for j in i:
            print(j)
        print()

    # for 1 process
    process_func(file_list, indices[0], output_dir)

