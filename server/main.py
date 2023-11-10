from flask import Flask, request
from unstructured.partition.pdf import partition_pdf
import tempfile
import os

from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage, SystemMessage
from server.image_tools import is_base64, split_image_text_types

from server.util import categorize_pdf_elements, prompt_func

vectorstore = Chroma(
    collection_name="cyrus_rag_clip", embedding_function=OpenCLIPEmbeddings()
)

retriever = vectorstore.as_retriever()

model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)
app = Flask(__name__)


@app.route("/")
def health():
    return "<p>up & running 🚀</p>"


@app.route("/process", methods=["POST"])
def process_pdf():
    file = request.files["file"]
    classes = request.form["classes"]

    with tempfile.TemporaryDirectory() as temp_dir:
        file.save(os.path.join(temp_dir, file.filename))
        file_path = os.path.join(temp_dir, file.filename)

        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=temp_dir,
        )

        # investigate this
        tables, texts = categorize_pdf_elements(raw_pdf_elements)

        image_uris = sorted(
            [
                os.path.join(temp_dir, image_name)
                for image_name in os.listdir(temp_dir)
                if image_name.endswith(".jpg")
            ]
        )

        vectorstore.add_images(uris=image_uris)

        vectorstore.add_texts(texts=texts)

        chain.invoke("do something super useful with the pdf")

    return "analyse"
