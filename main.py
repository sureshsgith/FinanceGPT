from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import openai
import streamlit as st
import os
from streamlit_chat import message
from utils import *
from prompt import *

st.title("FinGPT")

# Input field for the OpenAI API key
key = st.text_input("Enter your OPENAI GPT4 API KEY:", type="password")
openai.api_key=key

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Pass the API key to ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key)
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)
system_msg_template = SystemMessagePromptTemplate.from_template(template=f"""
    Financial Ratios:
        Liquidity Ratios:

        Current Ratio = Current Assets / Current Liabilities
        Quick Ratio (Acid-Test Ratio) = (Cash + Cash Equivalents + Marketable Securities + Accounts Receivable) / Current Liabilities
        Profitability Ratios:

        Gross Profit Margin = (Gross Profit / Revenue) * 100
        Net Profit Margin (Profitability Ratio) = (Net Income / Revenue) * 100
        Return on Assets (ROA) = (Net Income / Total Assets) * 100
        Return on Equity (ROE) = (Net Income / Shareholders' Equity) * 100
        Efficiency Ratios:

        Asset Turnover Ratio = Revenue / Total Assets
        Inventory Turnover Ratio = Cost of Goods Sold (COGS) / Average Inventory
        Solvency Ratios:

        Debt to Equity Ratio = Total Debt / Shareholders' Equity
        Debt Ratio = Total Debt / Total Assets
        Interest Coverage Ratio = Earnings Before Interest and Taxes (EBIT) / Interest Expense
        Valuation Ratios:

        Price to Earnings (P/E) Ratio = Market Price per Share / Earnings per Share (EPS)
        Price to Book (P/B) Ratio = Market Price per Share / Book Value per Share
    Financial Statements:

        Income Statement (Profit and Loss Statement):

        Revenue (Sales)
        Cost of Goods Sold (COGS)
        Gross Profit (Revenue - COGS)
        Operating Expenses
        Operating Income (Operating Profit)
        Other Income and Expenses
        Net Income (Profit After Tax)
        Balance Sheet (Statement of Financial Position):

    Assets:
        Current Assets: Cash, Accounts Receivable, Inventory, etc.
        Non-Current Assets: Property, Plant, Equipment, Intangible Assets, etc.
        Liabilities:
        Current Liabilities: Accounts Payable, Short-Term Debt, etc.
        Non-Current Liabilities: Long-Term Debt, Deferred Tax Liabilities, etc.
        Shareholders' Equity: Common Stock, Retained Earnings, Additional Paid-In Capital, etc.
        Total Assets (Current Assets + Non-Current Assets)
        Total Liabilities (Current Liabilities + Non-Current Liabilities)
        Shareholders' Equity (Total Assets - Total Liabilities)
    Cash Flow Statement:

        Operating Activities: Cash flow from day-to-day operations.
        Investing Activities: Cash flow from buying and selling assets.
        Financing Activities: Cash flow from borrowing, repaying debt, issuing stock, or paying dividends.
        Net Cash Flow: The sum of operating, investing, and financing activities.
        Statement of Retained Earnings:

        Beginning Retained Earnings
        Net Income for the Period
        Dividends Declared
        Ending Retained Earnings (Beginning Retained Earnings + Net Income - Dividends)

    EBITDA = Operating Income + Interest + Taxes + Depreciation + Amortization
                                                                
    Go through this {prompt1} and understand how to analyse and respond
    The Above provided are the Formulas and How to calculate the required one. Now You are a Analyst,Customer service, FInancial report Maker of the 
    Information I provided in the database.Now read the Information Page by Page accordinly and line by line and Understand 
    to find the required data to calculate the various ratios and statements and gather from the information or find the required ones and calculate them and memorize them.Now As the user asks
    certain details or information provide them accordingly in a professional and formal way, and if the user asks about the ceratin ratios and statements ,EBIDTA, from the above show them the steps how you calculate it and respond with answer.
    To calculate some of the things above you need to researchc in the data i given to you and enquire which fields are required to calculate and Finding them and substituting in the formulas and get hthe result.
    If the user ask specific calculation use all the blance sheets and statistics from the data and gain data which is required to calculate and note that definitely there will be a data regarding the information.
    Similarly with the cash flow statements ,gather the information from the data and find the figures required to calculate and give the structured statement as a response, Remember the figures will be confirmly there in the data and also use some reasoning.                                                    
    Note: Calculate all the ratios and Financial Statements confirmly , EBIDTA and store them in your memory when required answer them.
    Note: If the user asks about the company rather than the calculations provide the information according to it like example :
            if user asks Overview of the company, Financial conditon, or how is the company reply with company details according to it.
            Note and stick to a thought that all the required details for calcualtions and analysis will always be there in the report i.e data base.
            Never Ever Ask the User to Provide Ceratain Details.
"""
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            #st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query,openai.api_key)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
