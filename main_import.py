import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import tools_VertexAI_Search
import tools_DF
from LLM2 import llm_interaction
# ----------------------------------------------------------------------------------------------------------------------
service_account_file = './secrets/GL/ml-ops-poc-695-331cbd915e34.json'
# ----------------------------------------------------------------------------------------------------------------------
def send_message_to_chat(room_id, message_text):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
    credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=['https://www.googleapis.com/auth/chat.bot'])
    service = build('chat', 'v1', credentials=credentials)
    room_uri = f'https://chat.googleapis.com/v1/spaces/{room_id}/messages'
    message_body = {'text': message_text}
    response = service.spaces().messages().create(parent=room_uri, body=message_body).execute()
    print("Message sent to Google Chat room:", response)
# ----------------------------------------------------------------------------------------------------------------------
def run():
    Vector_Searcher.table_name = 'GL_catalog'
    df = Vector_Searcher.search_vector('Who are the leaders in GL?', select=['text', 'filename', 'url','score'], as_df=True,limit=4)
    if df.shape[0] ==0:return 'No results found'
    df = df[df['score']>0.75]
    if df.shape[0] ==0:return 'No relevant results found'
    df = df.sort_values(by='score', ascending=False)
    df['url'] = df['url'].apply(lambda x: Vector_Searcher.generate_signed_url(x, duration_sec=300))
    df['text'] = df['text'].apply(lambda x: Vector_Searcher.summarize(x))
    print(tools_DF.prettify(df, showindex=False))
    return
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    Vector_Searcher = tools_VertexAI_Search.VertexAI_Search('./secrets/GL/private_config_GCP.yaml','./secrets/GL/ml-ops-poc-695-331cbd915e34.json')
    # Vector_Searcher.cleanup_bucket(Vector_Searcher.bucket_name)
    # Vector_Searcher.add_book(table_name='GL_catalog',filename_in='./data/ex_datasets/pdf_GL/1B7mlWgckxEwVfXb_43m6Ci51I87gBkk_Kzjgj273Ef4.pdf')
    # Vector_Searcher.add_book(table_name='GL_catalog',filename_in='./data/ex_datasets/pdf_GL/1SYSoYgeuhGaG-ZVgOT7x-AJFPNyMyk64BzUglKaUFpw.pdf')

    #Vector_Searcher_GCP.add_book(table_name='Godfather',filename_in='./data/ex_datasets/pdf_books/Godfather.pdf',chunk_size=2000)

    #send_message_to_chat('YOUR_ROOM_ID', 'Hello, this is a test message!')
    #print(Vector_Searcher_GCP.generate_signed_url('gs://gl_catalog2/839a486d-dc60-477f-95f5-fc71eb9ebb85.png'))

    run()