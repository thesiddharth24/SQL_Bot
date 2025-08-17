
from prompts.schema import schema_table
from prompts.few_shots import fewShots
from llm.chatModels import get_chat_model as get_llm
from prompts.sql_query_make_prompt import query_prompt
from langchain_core.runnables import RunnableLambda , RunnableBranch
from Runnables.parse_llm_create_sql_fxn import parse_llm_output_fn
from Runnables.run_sql_query import execute_sql_query
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

question = "Delete all products in the 'Electronics' category."

parser = StrOutputParser()

llm = get_llm(temperature=0.0)



# Wrap into RunnableLambda
parse_llm_output = RunnableLambda(parse_llm_output_fn)

chain = query_prompt | llm | parse_llm_output


run_sql_query = RunnableLambda(execute_sql_query)

on_error_while_executing_query = RunnableLambda(lambda d: f"Error from DB while executing SQL query: {d.get('error', 'unknown error')}")
# Simple prompt template
results_to_text_prompt = PromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Here are the results of an SQL query:\n\n{result}\n\n"
    "Please summarize these results in plain English for the user."
)

branch2 = RunnableBranch(
    (lambda x: x['status'] == "ok", itemgetter("result") | results_to_text_prompt | llm | parser ),
    (lambda x: x['status'] == "error", on_error_while_executing_query ),
    RunnableLambda(lambda d: f"Unexpected status: {d.get('status')}")
)

branch = RunnableBranch(
    (lambda x: x["success"] == True ,  itemgetter("result")  | run_sql_query | branch2),
    lambda x: "Failed to generate SQL query"
)



final_chain = chain | branch

res = final_chain.invoke({
    "schema_table": schema_table,
    "few_shots": fewShots,
    "question": question
})



print(res)

