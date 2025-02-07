from pydantic import BaseModel, conlist
from typing import List, Optional
# ======================================================================
# Define validation schemas for API
# ======================================================================

# Each one of the entities to predict
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
    
class ReloadQuery(BaseModel):
    """
    When reloading a model we need a new config dictionary
    """
    model_id: str
    load_vino: bool
    load_tokenizer: bool
    vino_file: str
    
class ResponseUpdate(BaseModel):
    """
    When reloading a model we return a msg
    """
    message: str
    
class ConfigResponse(BaseModel):
    API_KEY: Optional[str] = None 
    API_INTERNAL_URL: Optional[str] = None
    MODEL_ID: str 
    LOAD_VINO: bool 
    LOAD_TOKENIZER: bool 
    VINO_FILE: str