from langchain.chains import create_extraction_chain

# Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}

# Input
input = """Alex is 5 feet tall. Claudia is 1 feet taller than Alex and jumps higher than him. Claudia is a brunette and Alex is blonde."""



from langchain_experimental.llms.ollama_functions import OllamaFunctions


import os

import dotenv

dotenv.load_dotenv()

 
OLLMA_BASE_URL = os.getenv("OLLMA_BASE_URL")


# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
model = llm = OllamaFunctions(
    model="mistral:instruct",
    base_url= OLLMA_BASE_URL
    )

# model = OllamaFunctions(model="mistral")

# Run chain
# llm = OllamaFunctions(model="mistral:instruct", temperature=0)
chain = create_extraction_chain(schema, llm)
output = chain.run(input)
x = 0