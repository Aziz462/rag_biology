from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from microservices.s3worker import S3WorkerParser
from microservices.vectorStoreWorker import vectorStoreWorker 
from microservices.llamaWork import LLMGenerator, APILLMGenerator

class Orchestrator:
    def __init__(self):
        # s3 for downloading pubmed files
        self.s3 = S3WorkerParser()
        self.embedderName = None
        self.crossEncoderName = None
        self.pathFile = "microservices/pmcIDs.txt"
        
        # initializations for the retriaval
        self.vectorStoreCreator = vectorStoreWorker()
        self.LLM = None
        
    def initVectorStore(self, embedderName, crossEncoderName):
        if not self.embedderName:
            self.embedderName = embedderName
        if not self.crossEncoderName:
            self.crossEncoderName = crossEncoderName
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedderName,
            model_kwargs={"trust_remote_code": True}
        )
        self.reranker = CrossEncoder(self.crossEncoderName)

        self.vectorstore = FAISS.load_local("./vectorstore_faiss", embeddings=self.embedder, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 50})
        print("initialized the vectorstore!")
    
    def recreateVectorStore(self):
        self.vectorStoreCreator.createVectorStore()
        self.initVectorStore(self.embedderName, self.crossEncoderName)

    def initHFLLM(self, model_id):
        self.LLM = LLMGenerator(model_id)
    
    def initAPILLM(self, api_key, model_name, base_url=None):
        self.LLM = APILLMGenerator(api_key, model_name, base_url)


    def lookUp(self, query):
        chunks = self.retriever.invoke(query)
        pairs = [(query, doc.page_content) for doc in chunks]
        scores = self.reranker.predict(pairs)
        
        # Zip chunks and scores, then sort
        reranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return reranked
    
    def addEntryToVectorStore(self, pmcID):
        if not self.s3.downloadSingleID(pmcID):
            print("couldn't download the file!")
            return False
        
        with open(self.pathFile, "a") as f:
            f.write(f"\n{pmcID}")


        file_path = f"output/{pmcID}.md"

        self.vectorStoreCreator.addSingleEntry(file_path)
        self.initVectorStore(self.embedderName, self.crossEncoderName)
        print("success!")
        return True
    
    def downloadFilesFromS3(self):
        self.s3.downloadByIDs(self.pathFile)

    def generateResponse(self, prompt, history, use_rag=False):
        context = "no context given"
        if use_rag:
            reranked_results = self.lookUp(prompt)
            # extract top 3 chunks for context window
            context_chunks = []
            for i, (doc, score) in enumerate(reranked_results[:3]):
                source = doc.metadata.get("source", "Unknown")
                context_chunks.append(f"Source [{source}] (Score: {score:.4f}):\n{doc.page_content}")
            
            context = "\n\n".join(context_chunks)
            print(f"Retrieved context from: {[doc.metadata.get('source') for doc, _ in reranked_results[:3]]}")
        
        if self.LLM:
            return self.LLM.generate_text(history, context)
        else:
            return "Error: LLM not initialized."