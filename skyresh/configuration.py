from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    API_KEY : str = Field(default="helloworld", env="API_KEY")
    MODEL_ID: str = Field(default="juampahc/gliner_multi-v2.1-openvino", env="MODEL_ID")
    LOAD_VINO: bool = Field(default=True, env="LOAD_VINO")
    LOAD_TOKENIZER: bool = Field(default=True, env="LOAD_TOKENIZER")
    VINO_FILE: str = Field(default="model_fp16.xml", env="VINO_FILE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

def get_settings():
    return Settings()
