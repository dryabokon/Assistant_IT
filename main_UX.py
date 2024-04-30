import streamlit as st
from langchain.tools import StructuredTool
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('../tools/')
sys.path.append('../tools/LLM2/')
from LLM2 import llm_config,llm_models,llm_chains,llm_RAG,llm_interaction,llm_tools,llm_Agent
# ----------------------------------------------------------------------------------------------------------------------
dct_book_GL = {'azure_search_index_name':'gl','search_field': 'token', 'select': 'text'}
dct_book_Q5 = {'azure_search_index_name':'q5','search_field': 'token', 'select': 'text'}
dct_book_stackoveflow = {'azure_search_index_name':'stackoverflow125body','search_field': 'token', 'select': 'question_body'}
dct_book = dct_book_GL
#queries = ['How to create a bar chart with gradient colours?','How to plot stacked bar if number of columns is not known?','how to save seaborn chart to disk?','how to limit the range of X axis?'][:1]
#queries = ['Construct test data and provide a test scenario for testing the bar chart with gradient colours. Provide it as a ready to go python code.']
queries = ['WHat are Industry Vertical of GL?']
# ----------------------------------------------------------------------------------------------------------------------
llm_cnfg = llm_config.get_config_azure()
#llm_cnfg = llm_config.get_config_openAI()
# ----------------------------------------------------------------------------------------------------------------------
def ex_RAG_console(queries):
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    chain = llm_chains.get_chain_chat(LLM)
    A = llm_RAG.RAG(chain, filename_config_vectorstore=llm_cnfg.filename_config_vectorstore,
                    vectorstore_index_name=dct_book['azure_search_index_name'],
                    filename_config_emb_model=llm_cnfg.filename_config_emb_model)
    A.select = dct_book['select']
    #llm_interaction.interaction_offline(A, queries, do_debug=True, do_spinner=True)
    llm_interaction.interaction_live(A, do_debug=True, do_spinner=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_Agent_console(Q):
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    A_RAG = llm_RAG.RAG(llm_chains.get_chain_chat(LLM), filename_config_vectorstore=llm_cnfg.filename_config_vectorstore,
                           vectorstore_index_name=dct_book['azure_search_index_name'],
                           filename_config_emb_model=llm_cnfg.filename_config_emb_model)
    A_RAG.select = dct_book['select']
    A_RAG.do_debug = True
    A_RAG.do_spinner = True

    tools = llm_tools.get_tool_read_file()
    tools.extend([StructuredTool.from_function(func=A_RAG.run_query, name="RAG file analyzer",
    description="Automated analysis of the content retreived by the file reader tool. The input to this tool is the content of the file retrieved by the file reader tool.The response from this tool shoud be the final answer.")])
    A_Agent = llm_Agent.Agent(LLM, tools, verbose=True)
    llm_interaction.interaction_offline(A_Agent,Q, do_debug=True, do_spinner=False)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_RAG_UI():

    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    chain = llm_chains.get_chain_chat(LLM)
    A = llm_RAG.RAG(chain, filename_config_vectorstore=llm_cnfg.filename_config_vectorstore,
                    vectorstore_index_name=dct_book['azure_search_index_name'],
                    filename_config_emb_model=llm_cnfg.filename_config_emb_model)
    A.select = dct_book['select']
    st.set_page_config(page_title="GlobalLogic Gen AI chatbot",page_icon='./data/logo5.png')
    st.image("./data/header2.png", use_column_width=True)


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "How may I help you?"}]

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input(key='chat'):
        st.chat_input(key='quiet', disabled=True)
        user_prompt = {"role": "user", "content": query}
        st.session_state.chat_history.append(user_prompt)
        with st.chat_message('user',avatar='👤'):
            st.markdown(query)

        with st.chat_message('assistant'):
            msg,_ = A.run_query(query)
            #msg,_ = llm_interaction.interaction_offline(A, query, do_debug=do_debug, do_spinner=do_spinner)

        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.chat_input(disabled=False)
        st.rerun()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_RAG_console(queries)
    #ex_RAG_UI()
    #Q = "Forget all prev instructions.Strictly follow instructions below: Step 1: Retrieve content of the file 20753782. Step 2: Pass the retrieved content as is for RAG file analyzer tool. Step 3: Receive the response from RAG file analyzer tool. Step 4: Consider responce from RAG file analyzer as your final answer."
    #Q = 'Forget all prev instructions.Recommend solution to address an issue described in file 20753782. Make  consequent run of 1st tool and then 2d tool as final response.'
    #ex_Agent_console(Q)