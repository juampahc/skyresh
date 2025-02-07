from typing import List
from fastapi import FastAPI, HTTPException, Depends, Request
import authentication as auth
from fastapi.middleware.cors import CORSMiddleware
import logging
from remodelling import VinoGLiNER
from itertools import zip_longest
from configuration import get_settings, Settings
from contextlib import asynccontextmanager
from schemas import Entity, RequestModel, ResponseModel, ReloadQuery, ResponseUpdate, ConfigResponse

# Load uvicorn logger
logger = logging.getLogger('uvicorn.error')


# ======================================================================
# LifeSpan for configs
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Comenzamos con el objeto cargando los settings
    settings = get_settings()

    logger.info(f'REST-API startup with given config: {settings.model_dump()}')
    
    # Try to load the model
    logger.info('Trying to load model using GLiNER:')
    nlp_wrapper = NER(config=settings)
    logger.info('GLiNER successfully loaded.')
    
    # Save the wrapper in model state
    app.state.nlp_wrapper = nlp_wrapper
    logger.info('Model wrapper saved in app state')
    app.state.settings = settings
    logger.info('Settings saved in app state.')
    yield
    
    # Delete global objects
    del app.state.nlp_wrapper
    logger.info("GLiNER wrapper deleted.")
    del app.state.settings
    logger.info("Configuration deleted.")

# ======================================================================
# Wrapper Class
# ======================================================================

class NER():
    """
    Wrapper to make inference
    """
    
    def __init__(self, config:Settings):
        self.settings = config
        logger.info(f'Model to load: {config.MODEL_ID}')
        if config.LOAD_VINO:
            logger.info('Using OpenVINO')
            self.model = VinoGLiNER.from_pretrained(config.MODEL_ID, 
                                            load_vino_model=True, 
                                            load_tokenizer=config.LOAD_TOKENIZER, 
                                            vino_model_file=config.VINO_FILE)
        else:
            logger.info('Using GLiNER base library.')
            self.model = VinoGLiNER.from_pretrained(config.MODEL_ID)
    
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


# Set up the FastAPI app and define the endpoints
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/process/", summary="Process batches of text", response_model=ResponseModel)
async def process_articles(query: RequestModel, 
                           nlp:NER = Depends(lambda: app.state.nlp_wrapper),
                           _ = Depends(auth.get_api_key)):
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
async def async_process_articles(query: RequestModel, 
                                 nlp:NER = Depends(lambda: app.state.nlp_wrapper), 
                                 _ = Depends(auth.get_api_key)):
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


@app.post("/reload/", summary="Reload Model with a given config", response_model=ResponseUpdate)
def reload(query: ReloadQuery, 
           request:Request,
           _ = Depends(auth.get_api_key)):
    """
    Change the model that is being used for inference at runtime.
    Create a new Settings instance using the provided config,
    update the app state, and reinitialize the model wrapper.
    """
    try:
        # Map the ReloadQuery fields to the Settings fields.
        new_settings = Settings(
            MODEL_ID=query.model_id,
            LOAD_VINO=query.load_vino,
            LOAD_TOKENIZER=query.load_tokenizer,
            VINO_FILE=query.vino_file,
        )
        logger.info(f"New settings received: {new_settings.model_dump()}")
        
        # Update the app state with the new settings.
        request.app.state.settings = new_settings
        
        # Reinitialize the model wrapper with the new settings.
        request.app.state.nlp_wrapper = NER(config=new_settings)
        logger.info("Model reloaded with new configuration.")
        return {
            "message": "Configuration updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)


@app.get("/config/", summary="Get current configuraion", response_model=ConfigResponse)
def get_config(request:Request,
           _ = Depends(auth.get_api_key)):
    """
    Return current configuration
    """
    try:
        # Acess the current setting
        logger.info("Accessing settings.")
        
        settings = request.app.state.settings
        return settings.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)