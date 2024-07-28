# Patent-Claims-Topic-Modeling
Our goal is to better understand competition in the mobile communications sector by studying patent claims. We want to create a method that groups patent claims into clear topics and use this method in a simple interactive tool. This tool should let users pick how many groups they want and show the names and number of claims in each group.
For this project, you'll need to install the following dependencies:

## Python 3 Installations Needed:
###  Libraries:
- requests: To make HTTP requests for web scraping.
- beautifulsoup4: For parsing HTML and XML documents.
- nltk: Natural Language Toolkit for text processing.
- gensim: For topic modeling using Latent Dirichlet Allocation (LDA).
- pandas: For data manipulation and analysis.
- Flask: Web framework for building the interactive application.
- pyLDAvis: For visualizing the LDA topics.

# Running the Notebooks and Flask Application:
##### To ensure proper execution, follow the sequence below:

### Data-Gathering.ipynb:
- Run this notebook first to scrape patent claims data from various sources.
- Save the extracted data as pickle files in the same directory as the code.
### Patent-Claims-Data-Cleaning.ipynb:
- Proceed to run this notebook after Data-Gathering.ipynb.
- Clean the extracted patent claims data to prepare it for analysis.
-  Save any cleaned data or intermediate results as pickle files in the same directory.
### LDA.ipynb:
- Run this notebook after Patent-Claims-Data-Cleaning.ipynb.
- Build Latent Dirichlet Allocation (LDA) models to identify topics within the patent claims.
- Save any generated models or processed data as necessary.
### flaskApp.py:
- Lastly, run the Flask application to interactively explore the LDA topic modeling results.
- Ensure all necessary files, including pickle files generated from previous steps, are saved in the same directory as the code.

