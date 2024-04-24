import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import utils_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
index_name_stack_overflow = 'stackoverflow125body'
index_name_Q5 = 'q5'
IO = utils_IO.IO(index_name_Q5)
# ----------------------------------------------------------------------------------------------------------------------
def pipe_00_pdfs_to_json(folder_in,filename_json):
    filenames_pdf = tools_IO.get_filenames(folder_in,list_of_masks='*.pdf')

    list_of_dict = []
    for filename_pdf in filenames_pdf:
        docs_pages = PyPDFLoader(folder_in + filename_pdf).load_and_split()
        texts_pages = [text.page_content for text in docs_pages]

        docs_all = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=800,chunk_overlap=100).create_documents(texts_pages)
        texts_all = [t.page_content for t in docs_all]
        #texts_all = [re.sub(r'[^ a-zA-Z0-9\-_=]', '', text) for text in texts_all]
        list_of_dict+= [{'item_id':'%06d'%int(i),'text':t} for i,t in enumerate(texts_all)]

    with open(filename_json,'w',encoding='utf-8') as f:
            json.dump(list_of_dict, f,indent='\t')

    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_01_import_KB(filename,index_name,field_source):
    dct_data = json.load(open(filename, encoding="utf-8"))
    IO.tokenize_and_upload(dct_data, index_name, field_source=field_source)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_02_search(query,index_name,select=['question_id','question_title']):
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
    #pipe_00_pdfs_to_json('./data/ex_datasets/pdf_GL/',filename_json='./data/ex_datasets/GL_ML.json')
    #pipe_00_pdfs_to_json('./data/ex_datasets/pdf_files_manuals/Q5/',filename_json='./data/ex_datasets/Q5.json')
    #pipe_00_pdfs_to_json('./data/ex_datasets/pdf_books/',filename_json='./data/ex_datasets/GF.json')
    #pipe_01_import_KB(filename='./data/ex_datasets/stackoverflow.json', field_source='question_body')
    #pipe_01_import_KB(filename='./data/ex_datasets/Q5.json', index_name='q5',field_source='text')

    IO.drop_index('gl')
    pipe_01_import_KB(filename='./data/ex_datasets/GL_ML.json', index_name='gl',field_source='text')

    #pipe_02_search('How to plot stacked bar if number of columns is not known?',index_name=index_name_stack_overflow,select = ['question_id','question_title'])
    #pipe_02_search('What gas should I use?',index_name=index_name_Q5,select = ['text'])
    #pipe_03_prepare_files('./data/ex_datasets/stackoverflow.json',folder_out)


