from sqlalchemy import create_engine # tools to connect to a database
from sqlalchemy.orm import sessionmaker, DeclarativeBase # tools to manage sessions abd base class for defining database tables
from .config import settings # contains database URL

# Foundation for creating database table models
class Base(DeclarativeBase):
    pass

# takes a database url and creates a connection to database tables, creates make_engine func that sets up connection to database using URL
# creates engine obj to manage the database connection
# sets up SessionLocal , a tool for creating sessions for interacting with the database
def make_engine(url: str):
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=False, future=True, connect_args=connect_args)


engine = make_engine(settings.database_url) # create the db connection using the url from settings
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)  # sets up a factory to create db sessions which is a workspace for running database operations

