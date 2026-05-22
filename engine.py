import os
import re
import csv
import json
import math
import difflib
import numpy as np
import pypdf
import onnxruntime as ort
from tokenizers import Tokenizer

class AIResearcherEngine:
    def __init__(self, model_path="model_files/model.onnx"):
        """Initializes the 20MB local Deep Learning engine and short-term memory."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at '{model_path}'. Please download it first.")
            
        # 1. Boot up the local ONNX execution graph
        self.session = ort.InferenceSession(model_path)
        
        # 2. Setup Tokenizer and enable strict 512 memory truncation to prevent array crashes
        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.enable_truncation(max_length=512) 
        
        # 3. Initialize memory databases
        self.vector_db = []
        self.last_important_words = set() # Powers the short-term conversation memory

    def clear_database(self):
        """Wipes the database and amnesiac memory clean."""
        self.vector_db = []
        self.last_important_words = set()

    def add_manual_text(self, label, text_content):
        """Injects custom manual data via the frontend dashboard."""
        self._semantic_chunk_and_store(text_content, source_name=f"Manual Input ({label})")

    def parse_file(self, file_path):
        """Safely opens and extracts text from various file formats."""
        ext = os.path.splitext(file_path)[-1].lower()
        extracted_text = ""
        
        try:
            if ext == ".pdf":
                # 'rb' mode forces Windows to release the file lock so we can delete it later
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text: extracted_text += text + " "
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            elif ext == ".csv":
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    extracted_text = " ".join([", ".join(row) for row in reader])
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    extracted_text = json.dumps(data)
            else:
                return False
        except Exception as e:
            print(f"Error reading file: {e}")
            return False

        self._semantic_chunk_and_store(extracted_text, source_name=os.path.basename(file_path))
        return True

    def _clean_words(self, text):
        """Utility: Strips punctuation and returns a set of lowercase clean words."""
        # This prevents "word," from failing to match "word"
        return set(re.findall(r'\b\w+\b', text.lower()))

    def _semantic_chunk_and_store(self, raw_text, source_name):
        """Slices text by sentence and implements the Sliding Window overlap upgrade."""
        sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())
        current_chunk = []
        current_words = 0
        
        for sentence in sentences:
            words = sentence.split()
            # If adding this sentence pushes us over 40 words, save the current chunk
            if current_words + len(words) > 40 and current_chunk:
                self.vector_db.append({"source": source_name, "text": " ".join(current_chunk)})
                
                # UPGRADE 1: Sliding Window Memory
                # Keep the very last sentence and carry it over to the next chunk
                overlap_sentence = current_chunk[-1] 
                current_chunk = [overlap_sentence]
                current_words = len(overlap_sentence.split())
                
            current_chunk.append(sentence)
            current_words += len(words)
            
        # Append the final remaining piece
        if current_chunk:
            self.vector_db.append({"source": source_name, "text": " ".join(current_chunk)})

    def ask(self, question):
        """The Master Router: Handles TF-IDF Math, Fuzzy Matching, and Context Stuffing."""
        if not self.vector_db:
            return "System Memory is empty. Please upload a file or type manual data first."

        # Filter out filler stop words
        stop_words = {"i", "need", "to", "know", "about", "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "by", "for", "with", "of", "this", "that", "it", "can", "what", "who", "where", "when", "how", "why", "tell", "me", "show", "from"}
        
        raw_q_words = self._clean_words(question)
        q_words = raw_q_words - stop_words
        
        if not q_words:
            q_words = raw_q_words # Fallback if user only typed stop words

        # UPGRADE 3: Short-Term Conversation Memory
        # If question is short (e.g., "when did it release?"), combine with previous topic
        if len(q_words) <= 2 and self.last_important_words:
            q_words = q_words.union(self.last_important_words)
        else:
            self.last_important_words = q_words.copy() # Save for next time

        # Generate a clean master vocabulary of every word in our database
        all_db_words = set()
        for chunk in self.vector_db:
            all_db_words.update(self._clean_words(chunk["text"]))

        # UPGRADE 4: Fuzzy Matching (Typo Forgiveness)
        corrected_q_words = set()
        for qw in q_words:
            if qw not in all_db_words:
                # Find an 80% similar word in the database (e.g., 'kratso' -> 'kratos')
                closest = difflib.get_close_matches(qw, list(all_db_words), n=1, cutoff=0.8)
                if closest:
                    corrected_q_words.add(closest[0])
                else:
                    corrected_q_words.add(qw)
            else:
                corrected_q_words.add(qw)
                
        q_words = corrected_q_words 

        # UPGRADE 2: TF-IDF Rare Word Weighting
        word_weights = {}
        total_chunks = len(self.vector_db)
        for w in q_words:
            appearances = sum(1 for chunk in self.vector_db if w in self._clean_words(chunk["text"]))
            if appearances > 0:
                # Rare words get a high multiplier; common words get a low multiplier
                word_weights[w] = math.log(total_chunks / appearances) + 1.0 
            else:
                word_weights[w] = 1.0

        # Score every chunk in the database
        chunk_scores = []
        for chunk in self.vector_db:
            c_words = self._clean_words(chunk["text"])
            score = sum(word_weights[w] for w in q_words if w in c_words)
            if score > 0:
                chunk_scores.append({"score": score, "text": chunk["text"], "source": chunk["source"]})

        # Sort chunks from highest mathematical relevance to lowest
        chunk_scores = sorted(chunk_scores, key=lambda x: x["score"], reverse=True)

        # Threshold check: If the top score is too low, the data isn't here
        if not chunk_scores or chunk_scores[0]["score"] < 0.5:
            return "Fallback Answer: That topic was not located inside your loaded datasets."

# UPGRADE 5 FIXED: Iterative Confidence Scoring
        # Instead of stitching text together, we test the Top 3 chunks individually
        best_answer = ""
        highest_confidence = -9999
        best_source = "Unknown"
        best_context_fallback = ""

        # Loop through the top 3 highest-scoring paragraphs
        for chunk in chunk_scores[:3]:
            encodings = self.tokenizer.encode(question, chunk["text"])
            
            if len(encodings.ids) > 510:
                 encodings.ids = encodings.ids[:510]
                 encodings.attention_mask = encodings.attention_mask[:510]
                 encodings.type_ids = encodings.type_ids[:510]
                 
            input_ids = np.array([encodings.ids], dtype=np.int64)
            attention_mask = np.array([encodings.attention_mask], dtype=np.int64)
            token_type_ids = np.array([encodings.type_ids], dtype=np.int64)
            
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            outputs = self.session.run(None, onnx_inputs)
            start_logits, end_logits = outputs[0][0], outputs[1][0]
            
            start_idx = int(np.argmax(start_logits))
            end_idx = int(np.argmax(end_logits))
            
            # THE MAGIC: Calculate how confident the AI is in this specific answer
            confidence = start_logits[start_idx] + end_logits[end_idx]
            
            # If this answer has the highest confidence so far, save it!
            if confidence > highest_confidence:
                highest_confidence = confidence
                answer_tokens = encodings.ids[start_idx : end_idx + 1]
                best_answer = self.tokenizer.decode(answer_tokens)
                best_source = chunk["source"]
                best_context_fallback = chunk["text"]

        # UPGRADE 0: Smart Fallback using the highest-confidence chunk
        if not best_answer.strip() or "[CLS]" in best_answer or "[SEP]" in best_answer or len(best_answer) <= 1:
            return f"[Context Found in '{best_source}']: I couldn't pinpoint a short exact answer for that phrasing, but here is the highly relevant data I found:\n\n{best_context_fallback}"
            
        return f"[From Source: '{best_source}'] Answer: {best_answer.strip()}"