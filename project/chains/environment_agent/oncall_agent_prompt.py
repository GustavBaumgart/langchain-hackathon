from langchain.prompts import PromptTemplate

prompt_template = """You want to query and summarize the article.
Use the article to get a summary using the topic.
Then, write that command.
    Article: {article}
    Topic: {topic}
    Command:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["article", "topic"]
)