from langchain.schema.messages import HumanMessage


def concatenate_pdf_elements(raw_pdf_elements):
    elements = []
    for element in raw_pdf_elements:
        elements.append(str(element))
    return elements
