from langchain.llms import OpenAI

from project.chains.environment_agent.oncall_agent_chain import OncallChain

def generate_command(error, index, _llm = OpenAI(temperature=0)):
    docs = index.similarity_search(error, k=1)
    inputs = [{"article": doc.page_content, "topic": error} for doc in docs]
    chain = OncallChain.from_llm(_llm)
    return chain.apply(inputs)