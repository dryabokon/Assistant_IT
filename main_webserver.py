from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('tools/')
sys.path.append('tools/LLM2/')
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_config,llm_models,llm_chains,llm_RAG
# ----------------------------------------------------------------------------------------------------------------------
dct_book_GL = {'azure_search_index_name':'gl','search_field': 'token', 'select': 'text'}
dct_book = dct_book_GL
llm_cnfg = llm_config.get_config_azure()
# ----------------------------------------------------------------------------------------------------------------------
PROJECT_ID = 'ml-ops-poc-695'
REGION = 'us-central1'
credentials = service_account.Credentials.from_service_account_file('./secrets/ml-ops-poc-695-331cbd915e34.json')
aiplatform.init(credentials=credentials, project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=REGION)
# ----------------------------------------------------------------------------------------------------------------------
LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
chain = llm_chains.get_chain_chat(LLM)
A = llm_RAG.RAG(chain, filename_config_vectorstore=llm_cnfg.filename_config_vectorstore,vectorstore_index_name=dct_book['azure_search_index_name'],filename_config_emb_model=llm_cnfg.filename_config_emb_model)
A.select = dct_book['select']
# ----------------------------------------------------------------------------------------------------------------------
from flask import Flask, request
app = Flask(__name__)
# ----------------------------------------------------------------------------------------------------------------------
@app.route('/')
def respond():
    query = request.args.get('query', 'No query provided')

    res = query
    #print(query)
    res,texts = A.run_query(query)
    print(texts)
    return res
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

