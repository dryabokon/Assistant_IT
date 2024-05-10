import datetime
import os
import uuid
import pandas as pd
import yaml
#----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_interaction,llm_models,llm_chains
import tools_time_profiler
#----------------------------------------------------------------------------------------------------------------------
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import bigquery
from langchain.schema.document import Document
from langchain_google_community import BigQueryVectorSearch
from langchain.vectorstores.utils import DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings
#----------------------------------------------------------------------------------------------------------------------
class VertexAI_Search(object):
    def __init__(self,filename_config,service_account_file=None,table_name=None):
        with open(filename_config, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
        self.credentials = service_account.Credentials.from_service_account_file(service_account_file)

        self.bigqwery_client = bigquery.Client(project=self.config['GCP']['PROJECT_ID'], location=self.config['GCP']['REGION'])
        self.model_emb_langchain = VertexAIEmbeddings(model_name="textembedding-gecko@latest", project=self.config['GCP']['PROJECT_ID'])
        self.BQ_dataset = 'my_vector_store'
        self.table_name = table_name
        self.bucket_name = self.config['GCP']['BUCKET']
        self.storage_client = storage.Client(project=self.config['GCP']['PROJECT_ID'],credentials=self.credentials)
        self.bucket = self.storage_client.bucket(self.bucket_name,user_project=self.config['GCP']['PROJECT_ID'])

        self.LLM = llm_models.get_model(filename_config, model_type='QA')
        self.chain = llm_chains.get_chain_summary(self.LLM)

        self.TP = tools_time_profiler.Time_Profiler()
        print('VertexAI_Search initialized')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,text):
        res = self.model_emb_langchain.embed_query(text)
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def upload_blob(self,bucket_name, source_file_name, destination_blob_name):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        command = 'gsutil cp ' + source_file_name +'gs://' +bucket_name + '/' + destination_blob_name
        # print(command)
        # os.system(command)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_signed_url(self,gs_url,duration_sec=60):
        bucket_name = '/'.join(gs_url.split('gs://')[1].split('/')[:-1])
        destination_blob_name = gs_url.split('gs://')[1].split('/')[-1]

        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        url = blob.generate_signed_url(expiration=datetime.timedelta(seconds=duration_sec),method='GET')
        return url
# ----------------------------------------------------------------------------------------------------------------------
    def cleanup_bucket(self,bucket_name):
        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs()
        for blob in blobs:
            blob.delete()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def add_book(self,table_name,filename_in,chunk_size=2000,add_images=True):
        self.TP.tic('add_book')

        is_pdf = (filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0)

        if is_pdf and add_images:
            texts,images = llm_interaction.pdf_to_texts_and_images(filename_in,chunk_size=chunk_size)
            metadata_list_of_dict = []
            for i,img in enumerate(images):
                uuid_filename = str(uuid.uuid4()) + '.png'
                img.save(uuid_filename)

                self.upload_blob(self.config['GCP']['BUCKET'], uuid_filename, uuid_filename)
                os.remove(uuid_filename)
                metadata_list_of_dict.append({'filename':filename_in.split('/')[-1],'url':'gs://'+self.config['GCP']['BUCKET']+ '/' +uuid_filename})
        else:
            texts = llm_interaction.pdf_to_texts(filename_in,chunk_size=chunk_size) if is_pdf else llm_interaction.file_to_texts(filename_in,chunk_size=chunk_size)
            metadata_list_of_dict =[{'filename':filename_in.split('/')[-1]}]*len(texts)

        self.bigqwery_client.create_dataset(dataset=self.BQ_dataset, exists_ok=True)

        table_id = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"
        self.bigqwery_client.create_table(table_id, exists_ok=True)

        store = BigQueryVectorSearch(
            project_id=self.config['GCP']['PROJECT_ID'],
            dataset_name=self.BQ_dataset,
            table_name=table_name,
            location=self.config['GCP']['REGION'],
            embedding=self.model_emb_langchain,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        store.add_texts(texts,metadatas=metadata_list_of_dict)
        print('%d' % len(texts) + ' chunks of %d added to BigQuery table ' % chunk_size + table_name)
        self.TP.print_duration('add_book')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def tbl_exists(self,table_ref):
        try:
            self.bigqwery_client.get_table(table_ref)
            return True
        except:
            return False
# ----------------------------------------------------------------------------------------------------------------------
    def summarize(self,text):
        summary = self.chain.run(question='Do very brief summary', input_documents=[Document(page_content=text, metadata={})])
        return summary
# ----------------------------------------------------------------------------------------------------------------------
    def search_vector(self,query,field=None, select=None, as_df=False,limit=4):

        if not self.tbl_exists(self.config['GCP']['PROJECT_ID']+'.'+self.BQ_dataset+'.'+self.table_name):
            return pd.DataFrame([]) if as_df else ''

        store = BigQueryVectorSearch(
            project_id=self.config['GCP']['PROJECT_ID'],
            dataset_name=self.BQ_dataset,
            table_name=self.table_name,
            location=self.config['GCP']['REGION'],
            embedding=self.model_emb_langchain,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        #docs = store.similarity_search_by_vector(self.get_embedding(query), k=limit)
        docs = store.similarity_search_with_score(query, k=limit)

        list_of_texts = [{'text':doc[0].page_content,'score':doc[1]} for doc in docs]
        list_of_metadata = [doc[0].metadata for doc in docs]

        for d1,d2 in zip(list_of_texts,list_of_metadata):
            d1.update(d2)

        res = llm_interaction.from_list_of_dict(list_of_texts,select=select,as_df=as_df)
        return res
# ----------------------------------------------------------------------------------------------------------------------
