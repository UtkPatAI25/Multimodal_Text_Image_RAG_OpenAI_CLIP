# Multimodal Text-Image RAG using OpenAI CLIP

This project provides a **Multimodal Retrieval Augmented Generation (RAG)** pipeline on PDFs containing both text and images. It leverages **OpenAI's CLIP model** for unified embedding of both modalities, and integrates them into a single FAISS vector store for efficient retrieval. The pipeline enables querying of a PDF using text and retrieves the most relevant text chunks and images, which are then provided as context to **GPT-4 Vision (GPT-4V)** for answering questions.

---

## Features

- **Parse PDF documents** with both text and images.
- **Embed text and image data** into a unified vector space using CLIP.
- **Search and retrieve** the most relevant text and image chunks for a user query.
- **Present context to GPT-4 Vision** (OpenAI GPT-4V) for natural language answers with visual context.
- **Fully automated pipeline**: PDF → Embeddings → Retrieval → LLM Response.

---

## Step-by-Step Flow

<img width="757" height="675" alt="image" src="https://github.com/user-attachments/assets/82a14a25-302c-48ad-9b80-22609429a945" />


### 1. **PDF Parsing**
- The pipeline reads a PDF file and iterates through each page.
- **Text Extraction**: Extracts text from each page and splits it into manageable chunks.
- **Image Extraction**: Extracts all images from PDFs, converts them to PIL images, and stores them as base64 (for LLM input).

### 2. **Embedding**
- **CLIP Model**: Both text and images are encoded with the pre-trained OpenAI CLIP model.
- **Unified Embedding Space**: Text and image embeddings are normalized and stored together.

### 3. **Vector Store Creation**
- **FAISS Index**: All embeddings (text & image) are loaded into a FAISS vector store for fast similarity search.

### 4. **Query and Retrieval**
- A user's text query is embedded using CLIP.
- The vector store retrieves the top-k most similar text and image chunks.

### 5. **Multimodal Prompt Construction**
- The pipeline creates a structured prompt for GPT-4 Vision, including both retrieved text and images (as base64).

### 6. **LLM Response**
- The prompt is sent to GPT-4 Vision (GPT-4V) via OpenAI API.
- The model generates a response grounded in both the text and image context.

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
PyMuPDF==1.23.9
langchain-core>=0.1.0
langchain>=0.1.0
langchain-community>=0.0.30
transformers>=4.38.0
torch>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
openai>=1.0.0
python-dotenv>=1.0.0
faiss-cpu>=1.7.4
```

---

## API Keys & Environment Variables

You will need an **OpenAI API key** for both CLIP model and GPT-4 Vision access.

1. Create a `.env` file in your project root:

    ```
    OPENAI_API_KEY=sk-xxxxxx...
    ```

2. The notebook will load this automatically using `python-dotenv`.

**Note:** Make sure your OpenAI API key has access to GPT-4 Vision (GPT-4V). For public CLIP model, the HuggingFace hub will be used; no extra key is needed.

---

## How to Use

1. **Place your PDF in the `./Data/` folder** (e.g., `./Data/GenAI_Report_2023_011124.pdf`).

2. **Run the Notebook**:  
   Open `Multimodal_Text_Image_RAG_openai_CLIP.ipynb` and execute cells in order.

3. **Example Queries** (in code):
    - "What does the chart on page 1 show about revenue trends?"
    - "Summarize the main findings from the document"
    - "What visual elements are present in the document?"

4. **Main Pipeline**:
    ```python
    answer = multimodal_pdf_rag_pipeline("Your question here")
    print(answer)
    ```

---

## Code Structure Overview

- **PDF Processing**: Extracts text and images from each page.
- **Embedding Functions**: Uses CLIP for both text and images.
- **FAISS Vector Store**: Stores all embeddings for similarity search.
- **Retrieval**: Finds the top relevant chunks/images for a query.
- **Prompt Construction**: Prepares the multimodal prompt for GPT-4V.
- **LLM Invocation**: Sends prompt to GPT-4 Vision and returns the answer.

---

## Tips

- For large PDFs, ensure sufficient memory is available.
- If using a GPU, PyTorch and Transformers will use it automatically.
- You can adjust the number of retrieved items (`k` parameter) in `retrieve_multimodal`.

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI GPT-4 Vision](https://platform.openai.com/docs/guides/vision)
