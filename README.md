\# Local AI Researcher Engine (20MB)



An offline, local Retrieval-Augmented Generation (RAG) engine built from scratch. It features TF-IDF rare-word weighting, sliding-window chunking, fuzzy typo-matching, and iterative confidence scoring to extract exact answers from multiple PDFs simultaneously.



\## Setup Instructions



1\. \*\*Install Dependencies:\*\*

&#x20;  `pip install pypdf onnxruntime numpy tokenizers fastapi uvicorn python-multipart`



2\. \*\*Download the AI Engine (17.5 MB):\*\*

&#x20;  `python download\_model.py`



3\. \*\*Boot the Server:\*\*

&#x20;  `python -m uvicorn app:app --host 127.0.0.1 --port 8000`

&#x20;  

&#x20;  Open `http://127.0.0.1:8000` in your browser to access the drag-and-drop dashboard.

