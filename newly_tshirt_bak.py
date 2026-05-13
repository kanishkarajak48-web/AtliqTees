import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="SQL Query Assistant",
    page_icon="🔍",
    layout="wide"
)

import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
GOOGLE_API_KEY = "AIzaSyAQzTAMboUqies3rZeVQujI_1SvbSJHkAU"
db_user = os.getenv("DB_USER", "root")
db_password = os.getenv("DB_PASSWORD", "root")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "3306")
db_name = os.getenv("DB_NAME", "mystore")

# Create SQLAlchemy engine
db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

@st.cache_resource
def init_database():
    """Initialize database connection with caching."""
    try:
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        return None

@st.cache_resource
def init_llm():
    """Initialize the LLM with caching."""
    if not GOOGLE_API_KEY:
        return None
    return GoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0, 
        google_api_key=GOOGLE_API_KEY
    )

# Initialize connections after page config
def initialize_connections():
    """Initialize database and LLM connections with proper error handling."""
    # Check API key
    if not GOOGLE_API_KEY:
        st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
        return None, None
    
    # Initialize database
    db = init_database()
    if db is None:
        st.error("Could not connect to database. Please check your database configuration.")
        return None, None
    
    # Initialize LLM
    llm = init_llm()
    if llm is None:
        st.error("Could not initialize the AI model.")
        return None, None
    
    return db, llm

# Custom prompt template
prompt_template = """
You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run.

IMPORTANT RULES:
- Return ONLY the SQL query, nothing else
- Do not include any explanatory text, prefixes, or suffixes
- Do not include markdown formatting, backticks, or code blocks
- Do not include words like "SQLQuery:", "SQL:", "Query:", or "Here is the query:"
- Return only the raw SQL command that can be executed directly

Question: {input}

SQL Query:"""

class SQLOutputParser(BaseOutputParser):
    """Custom output parser for SQL queries."""
    def parse(self, text):
        """Parse the output text."""
        return text.strip()

def clean_sql_query(query):
    """Clean the SQL query by removing markdown formatting and unnecessary characters."""
    cleaned_query = query.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        'SQLQuery:', 'SQL Query:', 'Query:', 'SQL:', 'sql:', 
        'Here is the SQL query:', 'The SQL query is:', 'Answer:'
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned_query.startswith(prefix):
            cleaned_query = cleaned_query[len(prefix):].strip()
    
    # Remove markdown formatting
    if cleaned_query.startswith('```sql'):
        cleaned_query = cleaned_query[6:]
    elif cleaned_query.startswith('```'):
        cleaned_query = cleaned_query[3:]
    
    if cleaned_query.endswith('```'):
        cleaned_query = cleaned_query[:-3]
    
    # Remove any trailing semicolon and add it back (normalize)
    cleaned_query = cleaned_query.rstrip(';').strip()
    
    # Remove any extra whitespace and newlines
    import re
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
    
    return cleaned_query.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_sql_query(question, db, llm):
    """Generate an SQL query based on the given question to successfully execute on mysql server."""
    try:
        # Create a custom prompt for better control
        custom_prompt = PromptTemplate.from_template(prompt_template)
        
        # Get database schema information
        table_info = db.get_table_info()
        
        # Create the full prompt with schema information
        full_prompt = f"""
You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run.

Database Schema:
{table_info}

IMPORTANT RULES:
- Return ONLY the SQL query, nothing else
- Do not include any explanatory text, prefixes, or suffixes
- Do not include markdown formatting, backticks, or code blocks
- Do not include words like "SQLQuery:", "SQL:", "Query:", or "Here is the query:"
- Return only the raw SQL command that can be executed directly

Question: {question}
"""
        
        # Generate the query
        response = llm.invoke(full_prompt)
        return response
        
    except Exception as e:
        st.error(f"Error generating SQL query: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_response(query, result, question, llm):
    """Generate a natural language response based on the SQL query and its result."""
    try:
        response_prompt = PromptTemplate.from_template(
            """
            You are a helpful data analyst. Based on the SQL query and its results, provide a clear, conversational answer to the user's question.

            User's Question: {question}
            SQL Query Executed: {query}
            Query Results: {result}

            Instructions:
            - Provide a natural, conversational response
            - Include specific numbers and details from the results
            - Make it easy to understand for non-technical users
            - If the result is empty, explain that clearly
            - If there are multiple rows, summarize the key findings
            - Use bullet points or formatting when helpful for readability

            Natural Language Response:
            """
        )
        response_chain = response_prompt | llm | SQLOutputParser()
        return response_chain.invoke({"query": query, "result": result, "question": question})
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        raise

def format_result_display(result, question):
    """Format the raw result for better display."""
    if not result:
        return "No data found for your query."
    
    # Try to parse the result and format it nicely
    try:
        # If result is a simple count or single value
        if isinstance(result, str) and result.replace('.', '').replace(',', '').isdigit():
            return f"**Result:** {result}"
        
        # If result contains multiple lines (likely multiple records)
        lines = str(result).strip().split('\n')
        if len(lines) > 1:
            formatted = "**Results:**\n\n"
            for i, line in enumerate(lines[:10], 1):  # Show first 10 results
                if line.strip():
                    formatted += f"{i}. {line}\n"
            if len(lines) > 10:
                formatted += f"\n... and {len(lines) - 10} more results"
            return formatted
        else:
            return f"**Result:** {result}"
    except:
        return f"**Result:** {result}"

def get_response(question, db, llm):
    """Process the user's question and return a response."""
    try:
        # Generate SQL query
        with st.spinner("🤖 Generating SQL query..."):
            sql_query = generate_sql_query(question, db, llm)
        
        # Clean the query
        cleaned_sql_query = clean_sql_query(sql_query)
        
        # Display the generated SQL query
        with st.expander("🔍 Generated SQL Query", expanded=False):
            st.code(cleaned_sql_query, language="sql")
        
        # Execute the query
        with st.spinner("⚡ Executing query..."):
            result = db.run(cleaned_sql_query)
        
        # Display raw results
        with st.expander("📊 Raw Query Results", expanded=False):
            formatted_result = format_result_display(result, question)
            st.markdown(formatted_result)
        
        # Generate natural language response
        with st.spinner("💬 Generating natural language response..."):
            nlp_response = generate_response(cleaned_sql_query, result, question, llm)
        
        return nlp_response, result, cleaned_sql_query
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        st.error(error_msg)
        return error_msg, None, None

def main():
    st.title("🔍 SQL Query Assistant")
    st.markdown("Ask questions about your database in natural language!")
    
    # Initialize connections
    db, llm = initialize_connections()
    if db is None or llm is None:
        st.stop()
    
    # Sidebar with database info
    with st.sidebar:
        st.header("Database Info")
        st.write(f"**Host:** {db_host}:{db_port}")
        st.write(f"**Database:** {db_name}")
        st.write(f"**User:** {db_user}")
        
        # Test database connection
        if st.button("Test Connection"):
            try:
                tables = db.get_usable_table_names()
                st.success("✅ Database connected successfully!")
                st.write("**Available Tables:**")
                for table in tables:
                    st.write(f"- {table}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., Show me all products with price greater than 100"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    if ask_button or question:
        if question.strip():
            nlp_response, raw_result, sql_query = get_response(question, db, llm)
            
            if nlp_response and raw_result is not None:
                # Display the natural language response prominently
                st.subheader("💬 AI Response:")
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                {nlp_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Add some spacing
                st.markdown("---")
                
                # Show additional insights if available
                st.markdown("### 📈 Additional Insights")
                try:
                    # Try to provide some context about the data
                    if raw_result:
                        result_str = str(raw_result)
                        if result_str.replace('.', '').replace(',', '').isdigit():
                            num_value = float(result_str.replace(',', ''))
                            if num_value > 1000:
                                st.info(f"💡 This is a large number ({num_value:,.0f}). Consider if this aligns with your expectations.")
                            elif num_value == 0:
                                st.warning("⚠️ The result is 0. You might want to check if this is expected or try a different query.")
                    
                    # Show query performance info in a simple format
                    st.write("**Query Information:**")
                    st.write(f"✅ Query executed successfully on database: {db_name}")
                    
                except Exception as e:
                    st.write("Additional analysis not available.")
                
                # Option to ask follow-up questions
                st.markdown("### 🔄 Follow-up Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Show me more details", key="more_details"):
                        follow_up = f"Can you provide more detailed information about: {question}"
                        st.info(f"Try asking: '{follow_up}'")
                
                with col2:
                    if st.button("📈 Related statistics", key="stats"):
                        follow_up = f"What are some related statistics for: {question}"
                        st.info(f"Try asking: '{follow_up}'")
                
                with col3:
                    if st.button("🔍 Similar queries", key="similar"):
                        st.info("Try variations like 'Show top 10...', 'What is the average...', 'Count of...'")
            
        else:
            st.warning("⚠️ Please enter a question.")
    
    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("### 📊 Data Analysis Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Basic Queries:**
            - Show me all products
            - What is the total number of customers?
            - List all product categories
            - Show me recent orders
            
            **Statistical Queries:**
            - What is the average price of products?
            - How many orders were placed this month?
            - Which product category has the most items?
            - What is the total revenue from all orders?
            """)
        
        with col2:
            st.markdown("""
            **Advanced Queries:**
            - Show me the top 5 most expensive products
            - Which customers have placed the most orders?
            - List products with low stock (less than 20)
            - Show me orders with status 'Processing'
            
            **Business Intelligence:**
            - What is our best-selling product category?
            - Which state has the most customers?
            - Show me monthly sales trends
            - List customers who haven't ordered recently
            """)
    
    # Quick action buttons for common queries (outside expander)
    st.markdown("### 🚀 Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📦 Show All Products", key="quick_products"):
            st.session_state['auto_question'] = "Show me all products with their prices"
    
    with quick_col2:
        if st.button("👥 Customer Count", key="quick_customers"):
            st.session_state['auto_question'] = "How many customers do we have?"
    
    with quick_col3:
        if st.button("💰 Total Revenue", key="quick_revenue"):
            st.session_state['auto_question'] = "What is our total revenue from all orders?"
    
    # Handle auto questions
    if 'auto_question' in st.session_state:
        auto_q = st.session_state['auto_question']
        del st.session_state['auto_question']
        st.success(f"**Auto Query:** {auto_q}")
        nlp_response, raw_result, sql_query = get_response(auto_q, db, llm)
        if nlp_response:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            {nlp_response}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()