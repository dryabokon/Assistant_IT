import json
import numpy
import pandas as pd
import pdf2image
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# ----------------------------------------------------------------------------------------------------------------------
from vertexai.language_models import TextEmbeddingInput
from vertexai.preview.language_models import TextEmbeddingModel
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import utils_IO
import tools_image
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
index_name_stack_overflow = 'stackoverflow125body'
IO = utils_IO.IO('gl')
# ----------------------------------------------------------------------------------------------------------------------
model_embeddings = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
# ----------------------------------------------------------------------------------------------------------------------
def remove_stop_worlds(text):
    for w in ['Conﬁdential','confidential','Conﬁdential','Condential','condential']:
        text = text.replace(w,'')

    return text
# ----------------------------------------------------------------------------------------------------------------------
def pipe_00_pdfs_to_json(folder_in,filename_json,do_embedding=False):
    def embed_text(texts):return [embedding.values for embedding in model_embeddings.get_embeddings([TextEmbeddingInput(text, "RETRIEVAL_DOCUMEN") for text in texts])]

    filenames_pdf = tools_IO.get_filenames(folder_in,list_of_masks='*.pdf')
    chunk_size = 1000
    df_pages_chunked = pd.DataFrame()
    for filename_pdf in filenames_pdf:
        imgs_pages = pdf2image.convert_from_path(folder_in + filename_pdf)
        imgs_pages = [tools_image.smart_resize(numpy.array(im), target_image_height=200) for im in imgs_pages]
        base64b = [tools_image.encode_base64(im) for im in imgs_pages]
        base64s = [str(b)[2:-1] for b in base64b]

        docs_pages = PyPDFLoader(folder_in + filename_pdf).load_and_split()
        texts_pages = [text.page_content for text in docs_pages]
        texts_pages = [s.encode("ascii", "ignore").decode() for s in texts_pages]
        texts_pages = [remove_stop_worlds(t) for t in texts_pages]
        df_pages = pd.DataFrame({'doc_id':filename_pdf.split('.')[0],'page_num':[str(i) for i in range(len(texts_pages))],'text':texts_pages,'len':[len(t) for t in texts_pages],'base64':base64s})

        idx = df_pages['len'] < chunk_size
        df_pages_chunked = pd.concat([df_pages_chunked,pd.concat([df_pages.loc[idx,:]])])
        for i in numpy.where(~idx)[0]:
            txts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size,chunk_overlap=100).create_documents([df_pages.loc[i, 'text']])
            df = pd.DataFrame({'doc_id':df_pages['doc_id'].iloc[i],'page_num':df_pages['page_num'].loc[i],'text':[t.page_content for t in txts],'len':[len(t.page_content) for t in txts],'base64':df_pages['base64'].loc[i]})
            df_pages_chunked = pd.concat([df_pages_chunked,df])

    df_pages_chunked = df_pages_chunked.drop(columns=['len'])
    df_pages_chunked = tools_DF.add_column(df_pages_chunked, 'uuid', [uuid.uuid4().hex for i in range(df_pages_chunked.shape[0])])

    if do_embedding:
        df_pages_chunked['embedding'] = embed_text(df_pages_chunked['text'])

    with open(filename_json,'w',encoding='utf-8') as f:
            json.dump(df_pages_chunked.to_dict(orient='records'), f,indent='\t')

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
if __name__ == '__main__':

    pass

    #pipe_00_pdfs_to_json('./data/ex_datasets/pdf_GL/',filename_json='./data/ex_datasets/GL_ML.json',do_embedding=False)
    #pipe_00_pdfs_to_json('./data/ex_datasets/pdf_files_manuals/Q5/',filename_json='./data/ex_datasets/Q5.json')
    #pipe_00_pdfs_to_json('./data/ex_datasets/pdf_books/',filename_json='./data/ex_datasets/GF.json')
    #pipe_01_import_KB(filename='./data/ex_datasets/stackoverflow.json', field_source='question_body')
    #pipe_01_import_KB(filename='./data/ex_datasets/Q5.json', index_name='q5',field_source='text')

    # IO.drop_index('gl')
    # pipe_01_import_KB(filename='./data/ex_datasets/GL_ML.json', index_name='gl',field_source='text')

    #pipe_02_search('How to plot stacked bar if number of columns is not known?',index_name=index_name_stack_overflow,select = ['question_id','question_title'])
    #pipe_02_search('What are business verticals?',index_name='gl',select = ['text'])



