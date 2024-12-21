# Khởi tạo chatbot RAG & MongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://quynhbuidoi123:3aeXTpv5R7JQF9f6@chatbot-cluster.3sfw9.mongodb.net/?retryWrites=true&w=majority&appName=ChatBot-Cluster"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
  client.admin.command('ping')
  print("Thành công kết nối MongoDB. You successfully connected to MongoDB!")
except Exception as e:
  print(e)

from sentence_transformers import SentenceTransformer
# https://huggingface.co/thenlper/gte-large
embedding_model = SentenceTransformer("thenlper/gte-large") #Load model hình để embedded câu trực tiếp từ hugging face
def get_embedding(text):
  if not text.strip():
    print('Attempted to embed empty text')
    return []
  embedding = embedding_model.encode(text)
  return embedding.tolist()

def vector_search(user_query, collection):
  query_embedding = get_embedding(user_query) # Sinh ra embedding từ câu user nhập vào
  if query_embedding is None:
    return "Invalid query of embedding generation failed.."
  # Define vector search pipeline
  vector_search_stage = {
      "$vectorSearch":{
          "index": "vector_index_quynhne",    # Tên của chỉ mục vector search
          "queryVector": query_embedding,
          "path": "embedding",      # Tìm kiếm trong cột embedding
          "numCandidates": 150, # Số lượng các câu gần với câu ta tìm kiếm
          "limit": 4 # Số lượng câu trả về
      }
  }
  unset_stage = {
      "$unset": "embedding"   # $unset xoá embedding trong  kết quả tìm kiếm
  }
  # Thiết lặp thông tin trả về
  project_stage = {
      "$project": {
          "_id": 0,     #Không chưa id
          "fullplot": 1,  # Chứa fullplot
          "title": 1,     # Chứa title...
          "genres": 1,
          "score": {
              "$meta": "vectorSearchScore"    # Chứa thông tin độ tương đồng
            }
      }
  }
  pipeline = [vector_search_stage, unset_stage, project_stage]
  # Bat dau tim kiem
  results = collection.aggregate(pipeline)
  return list(results)

def get_search_result(query, collection):
  get_knowledge = vector_search(query, collection)
  search_result = ""
  for result in get_knowledge:
    search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"
  return search_result

from huggingface_hub import login
import streamlit as st
import os
from huggingface_hub import login
# HUGGINGFACE_API_KEY_NE  
api_key = st.secrets["HUGGINGFACE_API_KEY_NE"]
# Lưu API key vào biến môi trường để notebook_login tự động sử dụng
os.environ["HUGGINGFACE_API_KEY_NE"] = api_key
token = os.environ["HUGGINGFACE_API_KEY_NE"]
# Đăng nhập vào Hugging Face Hub
login(token=token)

# # Load model directly || Dùng Model gemma để trả lời câu hỏi từ thông tin vừa được truy vấn,,, có thẻ dùng phobert nếu là tiếng việt
# from transformers import AutoTokenizer, AutoModelForCausalLM
# # https://huggingface.co/google/gemma-2b-it
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# #model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")  #CPU
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto") #GPU
from transformers import pipeline
# Hàm để tải model, chỉ chạy một lần
# @st.cache_resource
# def load_pipeline():
#   return pipeline("text-generation", model="google/gemma-2b-it")
# pipe = load_pipeline()
pipe = pipeline("text-generation", model="google/gemma-2b-it")










# Khởi tạo web chat streamlit
st.title("ChatGPT clone")

# Tao lich su chat
if "messages_ne" not in st.session_state:
  st.session_state.messages_ne = []

# Hien thi noi dung chat tu lich su vao web
for message in st.session_state.messages_ne:
  with st.chat_message(message['role']):
    st.markdown(message["content"])

# Phan hoi lai dau vao cua nguoi dung!!
prompt = st.chat_input("Nhập câu hỏi vào đây i..")
if prompt:                                  # Kiểm tra input chat là không rổng !
  # Hien thi tin nhan nguoi dung (input chat)
  with st.chat_message("user"):
    st.markdown(prompt)
  # Them tin nhan nguoi dung voi doan chat
  st.session_state.messages_ne.append({"role": "user", "content":prompt})

  # Lây thông tin từ Câu hỏi người dùng nhập vào
  query = prompt
  source_information = get_search_result(query, client['database_demo']['table1'])
  # Nối thông tin vừa lấy với câu hỏi
  combined_information = f"Query {query}\nContinue to answer the query by using the Search Results: \n{source_information}"

  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #input_ids = tokenizer(combined_information, return_tensors="pt").to(device)
  ##input_ids = tokenizer(combined_information, return_tensors="pt")
  #response = model.generate(**input_ids, max_new_tokens=512)
  #response = (tokenizer.decode(response[0]))
  response = combined_information
  response = pipe(combined_information,max_new_tokens=512)[0]['generated_text']
  # Hien thi phan hoi tro ly vao doan chat
  with st.chat_message("assistant"):
    st.markdown(response)
  # Them phan hoi troly vao lich su
  st.session_state.messages_ne.append({"role":"assistant", "content":response})

  # response = f"Echo: {prompt}"
  # # Hien thi phan hoi tro ly vao doan chat
  # with st.chat_message("assistant"):
  #   st.markdown(response)
  # # Them phan hoi troly vao lich su
  # st.session_state.messages_ne.append({"role":"assistant", "content":response})

  # with st.chat_message("assistant"):
  #   stream = openai.chat.completions.create(
  #       model=st.session_state["openai_model"],
  #       messages=[
  #           {"role": m["role"], "content": m["content"]} for m in st.session_state.messages_ne
  #       ],
  #       stream=True,
  #   )
  #   response = st.write_stream(stream)
  # st.session_state.messages.append({"role": "assistant", "content": response})







