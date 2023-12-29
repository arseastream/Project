# Import packages
import streamlit as st
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import seaborn as sns
import zipfile
import shutil
pd.set_option('display.max_colwidth', 800)

# Load loan-level data
@st.cache_resource
def load_loan_data(file_path):
    extract_dir = 'zip'
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    pkl_file_name = os.path.basename(file_path).replace('.zip', '.pkl')
    pkl_file_path = os.path.join(extract_dir, pkl_file_name)
    df = pd.read_pickle(pkl_file_path)
    os.remove(pkl_file_path)
    shutil.rmtree(extract_dir)
    df['Date'] = pd.to_datetime(df['Date'])
    df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'])
    return df

# Load dictionary
@st.cache_resource
def load_dictionary(file_path):
    dictionary = pd.read_excel(file_path)
    new_rows = pd.DataFrame({
        'Field Name': ['LoanID', 'Date'],
        'Definition': ['Unique identifier for each loan', 'Observation month']
    })
    dictionary = pd.concat([new_rows, dictionary]).reset_index(drop=True)
    return dictionary

# Set up gpt
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = openai_api_key)

# Set primer to prepare gpt for queries
primer = """You are a helpful assistant. 
            I will ask you for python scripts. 
            These scripts will deal with a dataframe called df. Do not edit the original df dataframe.
            This dataframe has columns Date, LoanID, Prepayment, ChargeOff, and Loan Age, amongst other columns.
            Prepayment and ChargeOff are either 1 or 0.
            Only return the python script, do not return any text explanations.
            Do not return any imports, assume that I have already imported all necessary packages.
            CPR is calculated as CPR = 1 - (1 - HP) ^ 12, where HP equals the average of the Prepayment column.
            CDR is calculated as CDR = 1 - (1 - HC) ^ 12, where HC equals the average of the ChargeOff column.
            When displaying CDR or CPR in a plot or table, format the CDR or CPR as a percentage to two decimal points.
            There is a column called Model Prepayment that contains a model probability that Prepayment is 1.
            Model CPR can be calculated by calculating HP using the average of the Model Prepayment column instead.
            Do not use plt.yticks().
            There is a streamlit that already exists, all results will be printed to this streamlit.
            Make sure to use groupby if appropriate.
            Do not use df.groupby().mean().
            For groupby, use a list if you want to refer to multiple columns.
            Refer to matplotlib.ticker as mtick if you use it.
            Only run mean() on specific columns, because some columns in df are non-numeric.
            Do not call st.pyplot without an argument, this will be deprecated soon.
            If date is involved, do not show the time component of dates, show just the calendar date. Use dt.date after groupby's to do this.
            If you are asked to plot, create a line plot without markers, make sure it includes a title and axis names, and show the plot on the streamlit using st.pyplot.
            If you are asked to plot, include a light grey bar plot on the secondary axis of the record count."""

# Additional primer to be ended at the end of the prompt
prompt_addition = """"""

# Create streamlit app and take in queries
def main():

    # Load data only once, using the cached function
    df = load_loan_data('sbadata_dyn_small.zip')
    dictionary = load_dictionary('7a_504_foia-data-dictionary.xlsx')

    # Set streamlit title
    st.title("SBA 504 Data Analysis with GPT")

    # Put general description of app
    st.write("""This application can be used to query the SBA 504 histrical performance data. So far we only have data for originations
                since 2010. The underlying data is monthly dynamic data. Example queries include asking for historical CDR's and CPR's,
                restricting to different populations.""")

    # Create tabs in Streamlit
    tab1, tab2, tab3 = st.tabs(["Queries", "User Guide", "Dictionary"])

    # Queries tab
    with tab1:

        # Set up user input. Make sure it ends in a period because there will be more after
        user_prompt = st.text_area("Enter your prompt:", "Type your prompt here...")
        if not user_prompt.endswith('.'):
            user_prompt += '.'

        # If button is clicked
        if st.button("Submit"):
            
            # Set up counters so app tries request max_attempts, in case gpt returns bad code
            max_attempts = 5
            attempts = 0
            success = False

            # Keep trying until max attempts
            while attempts < max_attempts and not success:
                try:
                    # Make a request to the OpenAI API
                    response = make_api_call(user_prompt)
                    response = response.replace("```python", "")
                    response = response.replace("```", "")

                    # Execute the script
                    exec_globals = {'df': df, 'pd': pd, 'plt': plt, 'mtick': mtick, 'mpl': mpl, 'st': st, 'np': np, 
                                    'MaxNLocator': MaxNLocator, 'mdates': mdates, 'sns': sns}
                    exec(response, exec_globals)

                    # Print the script from GPT
                    with st.expander("Python Script"):
                        st.code(response, language='python')
                    
                    # Set success if no errors
                    success = True

                except Exception as e:
                    # st.error(f"An error occurred: {e}")
                    attempts += 1

            # Requests a different query if gpt keeps giving bad code
            if not success:
                st.write("Please try a different query.")

    # User Guide tab
    with tab2:

        # Write tips for best query writing
        st.markdown("""
        **Tips For Writing Prompts**
        1. Write in full sentances
        2. Use exact variable names and values from the dictionary  
        3. Be as specific in your request as you can
        4. Remember that the code is dealing with a table  
        5. If you don't get what you want initially, try resubmitting the query

        **Example Queries**
        1. Plot the CDR by Date
        2. Plot the CPR by Loan Age
        3. Plot the CPR by Loan Age for the different BusinessType's. Round Loan Age by 12.  
        4. Plot the CDR by Loan Age when the Date was between 2015 and 2018. Restrict to where the record count is greater than 15000. Please also plot the record count by Loan Age on the secondary axis as bars. 
        5. Plot the model vs actual CPR by Date
        """)

    # Dictionary tab
    with tab3:

        # Header
        st.header("Dictionary")

        # Write dictionary
        dictionary['Definition'] = dictionary['Definition'].str.replace('\n', '<br>', regex=False)
        html = dictionary.to_html(index=False, escape=False)
        html = html.replace('<thead>', '<thead style="text-align: left;">')
        html = html.replace('<th>', '<th style="text-align: left;">')
        st.markdown(html, unsafe_allow_html=True)

# Submit query to gpt
def make_api_call(user_prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": user_prompt + prompt_addition}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    main()