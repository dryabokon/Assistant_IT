import tools_Azure_Search
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
class IO:
    def __init__(self,index_name):
        self.C = tools_Azure_Search.Client_Search('./secrets/GL/private_config_azure_search.yaml',filename_config_emb_model='./secrets/GL/private_config_azure_embeddings.yaml',index_name=index_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def drop_index(self,index_name):
        self.C.search_index_client.delete_index(index_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def upload(self,docs,new_index_name):
        if new_index_name not in self.C.get_indices():
            self.C.create_search_index(docs,field_embedding=None,index_name=new_index_name)

        self.C.search_client = self.C.get_search_client(new_index_name)
        self.C.upload_documents(docs)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_and_upload(self,docs,new_index_name,field_source='description',field_embedding='token'):
        docs_e = self.C.tokenize_documents(docs, field_source=field_source, field_embedding=field_embedding)
        if new_index_name not in self.C.get_indices():
            self.C.create_search_index(docs_e,'token',new_index_name)

        self.C.search_client = self.C.get_search_client(new_index_name)
        self.C.upload_documents(docs_e)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def search(self,query,index_name,select=None):
        self.C.search_client = self.C.get_search_client(index_name)
        df = self.C.search_semantic(query=query,select=select,as_df=True,limit=5)
        print('search_semantic')
        print(tools_DF.prettify(df, showheader=True, showindex=False))

        print('search_vector')
        df = self.C.search_vector(query=query,as_df=True, select=select,limit=5)
        print(tools_DF.prettify(df, showheader=True, showindex=False))

        print('search_hybrid')
        df = self.C.search_hybrid(query=query, as_df=True, select=select, limit=5)
        print(tools_DF.prettify(df, showheader=True, showindex=False))

        return
# ----------------------------------------------------------------------------------------------------------------------