import json
# ----------------------------------------------------------------------------------------------------------------------
import utils_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
index_name = 'stackoverflow125body'
IO = utils_IO.IO(index_name)
# ----------------------------------------------------------------------------------------------------------------------
def pipe_01_import_KB(filename,field_source):
    IO.tokenize_and_upload(json.load(open(filename, encoding="utf-8")), index_name, field_source=field_source)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_02_search(query,select=['question_id','question_title']):
    IO.search(query,index_name,select=select)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_03_prepare_files(filename_in,folder_out):
    data = json.load(open(filename_in, encoding="utf-8"))
    for d in data:
        d['question_body']
        with open(folder_out+str(d['question_id'])+'.txt','w',encoding='utf-8') as f:
            f.write(d['question_body'])

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # pipe_01_import_KB(filename='./data/ex_datasets/stackoverflow.json', field_source='question_body')
    # pipe_02_search('How to plot stacked bar if number of columns is not known?',select = ['question_id','question_title'])
    pipe_03_prepare_files('./data/ex_datasets/stackoverflow.json',folder_out)


