import os
import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

class vectorStoreWorker:
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-en"):
        self.FOLDER_PATH = "./output"  # folder containing your .md files
        self.MODEL_NAME = model_name
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.DEVICE = "mps"
            
        self.CHUNK_SIZE = 256  
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to(self.DEVICE)
        self.parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=8000, 
            chunk_overlap=0
        )

    def _get_embedder_wrapper(self):
        return HuggingFaceEmbeddings(
            model_name=self.MODEL_NAME,
            model_kwargs={"trust_remote_code": True, "device": self.DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )


    def lateChunk(self, large_texts: List[str]) -> Tuple[List[str], np.ndarray]:
        self.model.eval()
        small_chunks_text = []
        small_chunks_vectors = []

        inputs = self.tokenizer(
            large_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last_hidden_state for token embeddings
            all_token_embeddings = outputs.last_hidden_state

        # Move to CPU for processing loop
        input_ids_np = inputs['input_ids'].cpu().numpy()
        attention_mask_np = inputs['attention_mask'].cpu().numpy()

        for doc_idx in range(len(large_texts)):
            doc_ids = input_ids_np[doc_idx]
            doc_vectors = all_token_embeddings[doc_idx] # Keep on device for math
            doc_mask = attention_mask_np[doc_idx]
            actual_length = doc_mask.sum()

            for start in range(0, actual_length, self.CHUNK_SIZE):
                end = min(start + self.CHUNK_SIZE, actual_length)
                
                chunk_ids = doc_ids[start:end]
                text_chunk = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

                if not text_chunk.strip(): continue

                # mean pool vector
                vector_slice = doc_vectors[start:end]
                pooled_vector = torch.mean(vector_slice, dim=0)

                small_chunks_text.append(text_chunk)
                small_chunks_vectors.append(pooled_vector.cpu().numpy().astype("float32"))

        return small_chunks_text, np.array(small_chunks_vectors)

    def createParentChunksFromFile(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f_obj:
            content = f_obj.read()
            
            # split file into parent Chunks
            parent_chunks = self.parent_splitter.split_text(content)
            
            if not parent_chunks:
                print(f"Skipping empty file: {file_path}")
                return 

            # generate Late Chunking Embeddings
            # returns many small chunks (e.g., 256 tokens) per parent chunk
            file_texts, file_vectors = self.lateChunk(parent_chunks)
            
            # store results
            self.all_texts.extend(file_texts)
            self.all_vectors.extend(file_vectors)
            
            # add Metadata
            source_id = os.path.basename(file_path).replace(".md", "")
            for _ in range(len(file_texts)):
                self.all_metadatas.append({"source": source_id})
    
    def addSingleEntry(self, file_path):
        if not os.path.exists(file_path):
            return "No file found!"
        with open(file_path, "r", encoding="utf-8") as f_obj:
            content = f_obj.read()
            
            parent_chunks = self.parent_splitter.split_text(content)
            if not parent_chunks:
                return False
            
            new_texts, new_vectors = self.lateChunk(parent_chunks)

            text_embedding_pairs = list(zip(new_texts, new_vectors))
            source_id = os.path.basename(file_path).replace(".md", "")
            new_metadatas = [{"source": source_id} for _ in new_texts]
            
            vectorstore = FAISS.load_local(
                "./vectorstore_faiss", 
                self._get_embedder_wrapper(), 
                allow_dangerous_deserialization=True
            )
            
            vectorstore.add_embeddings(
                text_embeddings=text_embedding_pairs,
                metadatas=new_metadatas
            )
            
            vectorstore.save_local("./vectorstore_faiss")
            return True

    
    def createVectorStore(self):
        self.all_texts = []
        self.all_vectors = []
        self.all_metadatas = []

        print(f"Scanning folder: {self.FOLDER_PATH}")
        for filename in os.listdir(self.FOLDER_PATH):
            if filename.endswith(".md"):
                file_path = os.path.join(self.FOLDER_PATH, filename)
                print(f"Processing: {filename}...")
                try:
                    self.createParentChunksFromFile(file_path)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")


        if self.all_texts:
            print(f"Total chunks created: {len(self.all_texts)}")
            
            text_embedding_pairs = list(zip(self.all_texts, self.all_vectors))
            
            embedder_wrapper = HuggingFaceEmbeddings(
                model_name=self.MODEL_NAME,
                model_kwargs={"trust_remote_code": True, "device": self.DEVICE},
                encode_kwargs={"normalize_embeddings": True}
            )

            print("Building FAISS Index...")
            vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=embedder_wrapper,
                metadatas=self.all_metadatas
            )

            vectorstore.save_local("./vectorstore_faiss")
            print("success! Saved to ./vectorstore_faiss")
        else:
            print("No markdown files found or processed.")