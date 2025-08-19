from pydantic_settings import BaseSettings # lib to manage configuration settings { for database connection }

# Settings class stores and manage configuration details. allows in defining default settings
class Settings(BaseSettings):
    database_url: str = "sqlite:///./data.db"

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False

settings = Settings()

# basesettings - tool to create settings with automatic validation and support for loading values from env variables or files
# Settings class - points to local SQLite database file named 