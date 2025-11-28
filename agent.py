from langchain.tools import tool
from dependancies.pinecone_operations import PineconeOperations
from langchain.agents import create_agent

pinecone_ops = PineconeOperations()
vector_store = pinecone_ops.get_vector_store()

MODEL = "gpt-4o-mini"

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def get_answer(query):
    tools = [retrieve_context]
    # If desired, specify custom instructions
    prompt = (
        """You are an AI assistant specialized in Georgian legal content.
        You have access to a tool that retrieves context from a georgian legal content.
        Use the tool to help answer user queries."""
    )
    agent = create_agent(MODEL, tools, system_prompt=prompt)
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        message = event["messages"][-1]
    return message.content

if __name__ == "__main__":
    query = "1.	შესაძლოა თუ არა 16 წლის ასაკის ადამიანი იყოს დასაქმებული და თუ კი, რომელი მუხლით?"
    answer = get_answer(query)
    print("Final Answer:", answer)
