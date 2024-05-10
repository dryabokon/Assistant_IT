# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('tools/')
sys.path.append('tools/LLM2/')
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_console_color
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_config,llm_RAG,llm_chains
import tools_VertexAI_Search
# ----------------------------------------------------------------------------------------------------------------------
llm_cnfg = llm_config.get_config_GCP()
# ----------------------------------------------------------------------------------------------------------------------
Vector_Searcher = tools_VertexAI_Search.VertexAI_Search('./secrets/GL/private_config_GCP.yaml','./secrets/GL/ml-ops-poc-695-331cbd915e34.json')
LLM = Vector_Searcher.LLM
chain = llm_chains.get_chain_chat(LLM)
A = llm_RAG.RAG(chain, Vector_Searcher)
A.select = 'text'
# ----------------------------------------------------------------------------------------------------------------------
from flask import Flask, request
app = Flask(__name__)
# ----------------------------------------------------------------------------------------------------------------------
def do_search(table,query):
    print('Search:', query)
    Vector_Searcher.table_name = table
    df = Vector_Searcher.search_vector(query, select=['text', 'filename', 'url', 'score'],as_df=True, limit=6)
    if df.shape[0] == 0: return 'No results found'
    df = df[df['score'] > 0.75]
    if df.shape[0] == 0: return 'No relevant results found'
    df = df.sort_values(by='score', ascending=False)[:4]
    df['url'] = df['url'].apply(lambda x: Vector_Searcher.generate_signed_url(x, duration_sec=60))
    df['text'] = df['text'].apply(lambda x: Vector_Searcher.summarize(x))
    res = df.to_json(orient='records')
    print(tools_DF.prettify(df, showindex=False))
    return res
# ----------------------------------------------------------------------------------------------------------------------
def do_chat(table,query):
    print('Q:', query)
    A.Vector_Searcher.table_name = table
    res, texts = A.run_query(query)
    print(tools_console_color.apply_style(res, style='BLD'))
    print(tools_console_color.apply_style('\n'.join(texts), color='blk'))
    print(''.join(['-'] * 50))
    print('\n')
    return res
# ----------------------------------------------------------------------------------------------------------------------
@app.route('/')
def respond():
    #http://127.0.0.1:8080/?table=GL_catalog&query=who%20are%20GL%20leaders?
    table = request.args.get('table', 'No BQ table provided')  # "Godfather_2"
    query = request.args.get('query', 'No query provided')
    is_search_query = query.lower().startswith('search')
    if is_search_query:
        res = do_search(table, query)
    else:
        res = do_chat(table, query)

    return res
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

