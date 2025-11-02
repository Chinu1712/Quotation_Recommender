from dotenv import load_dotenv 
import streamlit as st 
import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
load_dotenv()
st.set_page_config(page_title="Best Quotation Recommender", page_icon="💬")
st.header("💬 Best Quotation Recommender")

# Tabs layout
tabs = st.tabs(["About", "Chatbot"])

# ===================================
# TAB 1: ABOUT
# ===================================
with tabs[0]:
    st.subheader("About this app")
    st.write("""
    Welcome to the **Best Quotation Recommender Chatbot!**

    This AI-powered application helps you analyze and compare quotation PDFs 
    to identify the most suitable one based on your query.

    **Key Features:**
    1. Upload one or multiple quotation PDF documents.
    2. Enter a query such as *"Which quotation offers better pricing and delivery terms?"*
    3. The AI analyzes the uploaded quotations and provides a context-based recommendation.
    4. Friendly, clear, and professional explanations.
    
    **Powered by:** LangChain ⚙️ | Groq AI ⚡ | Streamlit 💻
    """)

# ===================================
# TAB 2: CHATBOT
# ===================================
with tabs[1]:
    st.subheader("Quotation Recommender Chatbot")
    uploaded_files = st.file_uploader(
        "📄 Upload your PDF quotation documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    user_query = st.text_area("💬 Enter your query here:")

    if st.button("🔍 Get Best Quotation"):
        if not uploaded_files or user_query.strip() == "":
            st.warning("Please upload at least one PDF document and enter your query.")
        else:
            all_texts = []

            # Extract text from PDFs
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name

                loader = PyPDFLoader(temp_path)
                pages = loader.load_and_split()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = splitter.split_documents(pages)

                text = " ".join([doc.page_content for doc in docs])
                all_texts.append(text)

                st.success(f"✅ Loaded {len(pages)} pages and split into {len(docs)} chunks from '{uploaded_file.name}'")

            # Combine all document texts into one
            combined_text = "\n\n".join(all_texts)

            # Build the prompt
            prompt = PromptTemplate(
                template=(
                    "You are an expert quotation recommender.\n"
                    "Analyze the following quotations and answer the user query: {query}\n\n"
                    "Quotation Content:\n{document}\n\n"
                    "Guidelines:\n"
                    "1. Recommend the best quotation considering price, clarity, and terms.\n"
                    "2. If the query is unrelated, reply: 'I am  a helpful AI assistant specializing in quotations.'\n"
                    "3. Be polite, clear, and concise.\n"
                    "4. Base your reasoning strictly on the provided quotation text.\n"
                ),
                input_variables=["query", "document"]
            )

            # Load Groq model
            model = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="openai/gpt-oss-120b"
            )

            # Build the chain
            parser = StrOutputParser()
            quotation_chain = prompt | model | parser

            # Get the response
            response = quotation_chain.invoke({
                "query": user_query,
                "document": combined_text
            })

            # Display the output
            st.subheader("🧠 Best Quotation Recommendation:")
            st.write(response)
            
        



