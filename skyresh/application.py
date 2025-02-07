from typing import List
from fastapi import FastAPI, HTTPException, Depends
import authentication as auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import logging
from remodelling import VinoGLiNER
from itertools import zip_longest
from configuration import get_settings

# Load uvicorn logger
logger = logging.getLogger('uvicorn.error')

# ======================================================================
# Define validation schemas for API
# ======================================================================

# Each una of the entities to predict
class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float

# Schema for the response
class ResponseModel(BaseModel):
    # Conjunto de entidades
    class Batch(BaseModel):
        text: str
        ents: List[Entity] = []
    result: List[Batch]

# Schema for input data
class Document(BaseModel):
    text: str

# Batch of documents, part of the request
class RequestModel(BaseModel):
    """
    General class for the expected request.
    We assume that the same labels are applied to all the batch of
    texts.
    Max batch_size of 100
    Max number of entities of 25
    """
    documents: conlist(Document, min_length=1, max_length=100) # type: ignore
    # Etiquetas que debemos buscar
    labels: conlist(str, min_length=1, max_length=25)     # type: ignore
    threshold: float

# ======================================================================
# Wrapper Class
# ======================================================================

class NER():
    """
    Wrapper to make inference
    """
    settings = get_settings()
    def __init__(self):
        logger.info(f'Model to load: {NER.settings.MODEL_ID}')
        if NER.settings.LOAD_VINO:
            logger.info('Using OpenVINO')
            self.model = VinoGLiNER.from_pretrained(NER.settings.MODEL_ID, 
                                            load_vino_model=True, 
                                            load_tokenizer=NER.settings.LOAD_TOKENIZER, 
                                            vino_model_file=NER.settings.VINO_FILE)
        else:
            self.model = VinoGLiNER.from_pretrained(NER.settings.MODEL_ID)
    
    def inference(self, texts: list[str], labels:list[str], threshold:float) -> List[List[Entity]]:
        """
        sync call for inference
        """
        # Debemos extraer los textos
        return self.model.batch_predict_entities(texts,labels=labels, threshold=threshold)
    
    async def async_inference(self, texts: list[str], labels:list[str], threshold:float) -> List[List[Entity]]:
        """
        Async call for inference
        """
        # Debemos extraer los textos
        return await self.model.async_batch_predict_entities(texts,labels=labels, threshold=threshold)

# Try to load the model
logger.info('Trying to load model using GLiNER:')
nlp = NER()
logger.info('GLiNER successfully loaded.')

# Set up the FastAPI app and define the endpoints
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/process/", summary="Process batches of text", response_model=ResponseModel)
async def process_articles(query: RequestModel, _ = Depends(auth.get_api_key)):
    """
    Process a batch of articles and return the entities predicted by the
    given model. Each record in the data should have a key "text".
    """
    response_body = []
    try:
        docs = [document.text for document in query.documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    try:
        results = nlp.inference(docs, query.labels, query.threshold)
        # Creamos la estructura que devolvemos como respuesta
        # Para cada miembro del batch se devuelve un diccionario
        for element,result in zip_longest(docs,results, fillvalue=None):
            if element is None or result is None:
                logger.error('Something went wrong, missmatch size between batch prediction and input')
                continue
            response_body.append({"text":element,
                                  "ents":result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    if not response_body:
        raise HTTPException(status_code=519, detail="Something went *really* wrong.")
    return {"result": response_body}


@app.post("/async_process/", summary="Process batches of text asynchronous", response_model=ResponseModel)
async def async_process_articles(query: RequestModel, _ = Depends(auth.get_api_key)):
    """
    Process a batch of articles and return the entities predicted by the
    given model. Each record in the data should have a key "text".
    """
    response_body = []
    try:
        docs = [document.text for document in query.documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    try:
        results = await nlp.async_inference(docs, query.labels, query.threshold)
        # Creamos la estructura que devolvemos como respuesta
        # Para cada miembro del batch se devuelve un diccionario
        for element,result in zip_longest(docs,results, fillvalue=None):
            if element is None or result is None:
                logger.error('Something went wrong, missmatch size between batch prediction and input')
                continue
            response_body.append({"text":element,
                                  "ents":result})
    except Exception as e:
        if not nlp.settings.LOAD_VINO:
            raise HTTPException(status_code=500, detail='GLiNER base model not compatible with async.')
        else:
            raise HTTPException(status_code=500, detail=e)
    if not response_body:
        raise HTTPException(status_code=519, detail="Something went *really* wrong.")
    return {"result": response_body}