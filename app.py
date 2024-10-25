import streamlit as st
import os
import logging
import sqlparse
import json
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Load the .env file
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the database
engine = create_engine("sqlite:///data.db")
db = SQLDatabase(engine)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Get context from the database
context = db.get_context()

# Read example queries from JSON file
with open("examples.json", "r") as f:
    examples = json.load(f)

# Define the example prompt
example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

# Define the example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

# Define the few-shot prompt template with example selector
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run not including sql``` or any prefixes before the start of the query. Unless otherwise specified, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

# Initialize the query execution tool
execute_query = QuerySQLDataBaseTool(db=db)

# Initialize the query writing chain
write_query = create_sql_query_chain(llm, db, prompt) # prompt included in the chain

# Function to pretty print SQL query with indentation
def pretty_print_sql(query, indent="    "):
    formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
    indented_query = "\n".join(indent + line for line in formatted_query.splitlines())
    return indented_query

### VALID with logging query to object ###

# Configure logging
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{record.getMessage()}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Set custom formatter for all handlers
for handler in logging.root.handlers:
    handler.setFormatter(CustomFormatter())

# Set logging level for httpx to WARNING to suppress its INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Define the prompt template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# List to store logged queries
logged_queries = []

# Function to log the query
def log_query(query):
    # Temporarily disable logging
    logging.disable(logging.INFO)
    logger.info(f"Executing SQL Query: {query}")
    # Re-enable logging
    logging.disable(logging.NOTSET)
    logged_queries.append(query)  # Save the query to the list
    return query

# Define the chain
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | (log_query | execute_query)
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Function to invoke the chain
def invoke_chain(question):
    # Invoke the chain and capture the result
    result = chain.invoke({"question": question})
    # Format the logged query
    formatted_query = pretty_print_sql(logged_queries[-1])
    return result, formatted_query

# Streamlit app
st.markdown(
    """
    <style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("./images/wb_logo_01.png", caption=None, width=200, use_column_width=False)
st.title("Insights on Demand")

# Inject custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state for logging questions and answers
if 'qa_log' not in st.session_state:
    st.session_state.qa_log = []

# Create a form for user input
with st.form(key='query_form'):
    question = st.text_area("Ask a question:", height=150)
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if question:
        summarized_answer, formatted_query = invoke_chain(question)
        
        # Log the question and answer
        st.session_state.qa_log.append({
            "question": question,
            "query": formatted_query,
            "answer": summarized_answer
        })
        
        # Display the formatted query
        st.subheader("Executed Query:")
        st.code(formatted_query, language='sql')
        
        # Display the summarized answer
        st.subheader("Summarized Answer:")
        st.write(summarized_answer)
    else:
        st.write("Please enter a question.")

# Display the log of questions and answers on the left side
st.sidebar.title("Question and Answer Log")
for entry in st.session_state.qa_log:
    st.sidebar.subheader("Question:")
    st.sidebar.write(entry["question"])
    st.sidebar.subheader("Executed Query:")
    st.sidebar.code(entry["query"], language='sql')
    st.sidebar.subheader("Summarized Answer:")
    st.sidebar.write(entry["answer"])