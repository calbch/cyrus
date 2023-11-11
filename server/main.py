from flask import Flask, jsonify, request
from flask_cors import CORS
from unstructured.partition.pdf import partition_pdf
import tempfile
import os

from langchain.chat_models import ChatOpenAI

from server.classes import PdfSummary


from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from server.util import concatenate_pdf_elements

app = Flask(__name__)
CORS(app)


@app.route("/handshake")
def health():
    return "up & running 🚀"


@app.route("/process", methods=["POST"])
def process_pdf():
    print("Processing PDF request 🗃️")
    file = request.files["file"]
    classes = request.form["classes"]

    with tempfile.TemporaryDirectory() as temp_dir:
        file.save(os.path.join(temp_dir, file.filename))
        file_path = os.path.join(temp_dir, file.filename)

        raw_pdf_elements = partition_pdf(
            filename=file_path,
            strategy="fast",
        )

        elements = concatenate_pdf_elements(raw_pdf_elements)
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class algorithm for extracting information in structured formats. You know a lot about different classes at university and you are able to summarize and categorize complex texts.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: {input}",
                ),
                (
                    "human",
                    "Choose a classification from the following list: {classes}",
                ),
                (
                    "human",
                    "Make sure to answer in the correct format and dont repeat information in the summary that is present in title or classification.",
                ),
                (
                    "human",
                    "I am studying for finals and your summary is crucial for my success.",
                ),
            ]
        )

        runnable = create_structured_output_runnable(PdfSummary, llm, prompt)
        result = runnable.invoke({"input": elements, "classes": classes})

        return jsonify(
            {
                "title": result.title,
                "class": result.classification,
                "summary": result.summary,
            }
        )
