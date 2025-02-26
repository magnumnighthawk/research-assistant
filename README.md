# Research Assistant

This project is a research assistant tool that fetches research papers from ArXiv, summarizes them using OpenAI's GPT-4 model, and stores the summaries in a FAISS-based vector store for easy retrieval.

## Features

- Fetch research papers from ArXiv
- Summarize research papers using OpenAI's GPT-4 model
- Store summaries in a FAISS-based vector store
- Retrieve stored summaries using similarity search

## Requirements

- Python 3.7+
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/research_assistant.git
    cd research_assistant
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Create a .env file based on the .env.sample file and update it with your API keys:

    ```sh
    cp .env.sample .env
    ```

4. Update the .env file with your API keys:

    ```env
    OPENAI_API_KEY='your-openai-api-key'
    HUGGINGFACEHUB_API_TOKEN='your-huggingfacehub-api-token'
    LANGCHAIN_API_KEY='your-langchain-api-key'
    TAVILY_API_KEY='your-tavily-api-key'
    ANTHROPIC_API_KEY='your-anthropic-api-key'
    ```

## Usage

1. Run the main script:

    ```sh
    python main.py
    ```

2. Choose an option from the menu:

    - (1) Research a new topic: Fetch and summarize research papers from ArXiv.
    - (2) Search through saved summaries: Retrieve stored summaries using similarity search.
    - (3) Exit: Exit the program.

## Example

### Research a New Topic

1. Choose option (1) and enter a research topic (e.g., "machine learning").
2. The program will fetch research papers from ArXiv, summarize them, and store the summaries in the FAISS-based vector store.
3. The stored summaries will be ready for retrieval.

### Search Through Saved Summaries

1. Choose option (2) and enter a search query (e.g., "neural networks").
2. The program will retrieve and display the most relevant summaries from the vector store.

## Project Structure

- main.py: Main script that contains the core functionality.
- .env.sample: Sample environment file with placeholders for API keys.
- requirements.txt: List of required Python packages.
- faiss_index.bin: FAISS index file (ignored by Git).
- docstore.pkl: Docstore file (ignored by Git).
- index_to_docstore_id.pkl: Index to docstore ID mapping file (ignored by Git).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com/)
- [ArXiv](https://arxiv.org/)
- [FAISS](https://github.com/facebookresearch/faiss)