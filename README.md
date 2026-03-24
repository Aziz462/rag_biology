# rag_biology
This is a pet project rag-system designed for biology researches.
The main feature - is using AWS S3 to pull articles from PubMed by their PMCID and add them to the vectorstore right away.
# How to run this
1. download all the dependencies 

```pip install -r requirements.txt```

2. run the streamlit app

```streamlit run app.py```

3. you can then choose either running llm via some openai compatible api or run some model locally.
