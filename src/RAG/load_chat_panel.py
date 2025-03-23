import warnings
import textwrap
from uuid import uuid4

import panel as pn
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

from insuriq_utils import INSURIQ_CACHE
from insuriq_utils import download_olmo_model

warnings.filterwarnings("ignore")

# enable Panel extension
pn.extension()

# get the OLMo model path
model_path = download_olmo_model()

# set the Qdrant vector DB cache path and collection name
qdrant_path = INSURIQ_CACHE / "insurance_documents"
qdrant_collection = "insurance_documents"

# load the sentence-transformer model for embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# load the existing Qdrant vector store
db = Qdrant.from_existing_collection(
    collection_name=qdrant_collection, embedding=embedding, path=qdrant_path
)

# prompt template
input_prompt_template = textwrap.dedent(
    """\
You are a knowledgeable assistant specializing in health insurance. 
Please answer the question using the relevant information from the following context:

{context}

Question: {question}
"""
)

def get_chain(callback_handlers: list[BaseCallbackHandler], input_prompt_template: str):
    # 1. Set up the vector database retriever.
    retriever = db.as_retriever(
        callbacks=callback_handlers,
        search_type="mmr",  # Maximal Marginal Relevance for diverse results
        search_kwargs={"k": 2},  # Top 2 results
    )

    # 2. Setup the callback manager to stream outputs to panel
    callback_manager = CallbackManager(callback_handlers)

    # 3. Load the OLMo-7B-Instruct model using llama-cpp
    olmo = LlamaCpp(
        model_path=str(model_path),
        callback_manager=callback_manager,
        temperature=0.8,
        n_ctx=4096,
        max_tokens=512,
        verbose=False,
        echo=False,
    )

    # 4. Load the model's built-in tokenizer chat template and use jinja2 format
    prompt_template = PromptTemplate.from_template(
        template=olmo.client.metadata["tokenizer.chat_template"],
        template_format="jinja2",
        partial_variables={
            "add_generation_prompt": True,
            "eos_token": "<|endoftext|>",
        },
    )

    # 5. Customize the prompt structure to take in a context and question from the user
    transformed_prompt_template = PromptTemplate.from_template(
        prompt_template.partial(
            messages=[
                {
                    "role": "user",
                    "content": input_prompt_template,
                }
            ]
        ).format()
    )

    # 6. Helper function to format document contents into string
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # 7. Hook to pass retrieved documents to callback for Panel rendering
    def show_docs(docs):
        for callback_handler in callback_handlers:
            callback_handler.on_retriever_end(docs, run_id=uuid4())
        return docs

    # 8. Compose the full LangChain pipeline with document retrieval → prompt creation → model inference
    return (
        {
            "context": retriever | show_docs | format_docs,
            "question": RunnablePassthrough(),
        }
        | transformed_prompt_template
        | olmo
    )

# callback function that powers the Panel chat interface
async def callback(contents, user, instance):
    # 1. Create a Panel callback handler for Langchain
    callback_handler = pn.chat.langchain.PanelCallbackHandler(
        instance, user="OLMo", avatar="🌳"
    )

    # 2. Prevent double printing of final response
    callback_handler.on_llm_end = lambda response, *args, **kwargs: None

    # 3. Build the LangChain pipeline with the callback handler and prompt
    chain = get_chain(
        callback_handlers=[callback_handler],
        input_prompt_template=input_prompt_template,
    )

    # 4. Invoke the pipeline asynchronously using Panel's chat message
    _ = await chain.ainvoke(contents)

# initialize and serve the Panel chat interface
chat_interface = pn.chat.ChatInterface(callback=callback)
pn.serve({"/": chat_interface}, port=5006, websocket_origin="*", show=False)
