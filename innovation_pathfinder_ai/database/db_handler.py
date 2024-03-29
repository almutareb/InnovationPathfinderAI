from sqlmodel import SQLModel, create_engine, Session, select
from innovation_pathfinder_ai.database.schema import Sources
from innovation_pathfinder_ai.utils.logger import get_console_logger
import os
from dotenv import load_dotenv

load_dotenv()

sqlite_file_name = os.getenv('SOURCES_CACHE')

sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)

logger = get_console_logger("db_handler")

SQLModel.metadata.create_all(engine)


def read_one(hash_id: dict):
    with Session(engine) as session:
        statement = select(Sources).where(Sources.hash_id == hash_id)
        sources = session.exec(statement).first()
        return sources


def add_one(data: dict):
    with Session(engine) as session:
        if session.exec(
            select(Sources).where(Sources.hash_id == data.get("hash_id"))
        ).first():
            logger.warning(f"Item with hash_id {data.get('hash_id')} already exists")
            return None  # or raise an exception, or handle as needed
        sources = Sources(**data)
        session.add(sources)
        session.commit()
        session.refresh(sources)
        logger.info(f"Item with hash_id {data.get('hash_id')} added to the database")
        return sources


def update_one(hash_id: dict, data: dict):
    with Session(engine) as session:
        # Check if the item with the given hash_id exists
        sources = session.exec(
            select(Sources).where(Sources.hash_id == hash_id)
        ).first()
        if not sources:
            logger.warning(f"No item with hash_id {hash_id} found for update")
            return None  # or raise an exception, or handle as needed
        for key, value in data.items():
            setattr(sources, key, value)
        session.commit()
        logger.info(f"Item with hash_id {hash_id} updated in the database")
        return sources


def delete_one(id: int):
    with Session(engine) as session:
        # Check if the item with the given hash_id exists
        sources = session.exec(
            select(Sources).where(Sources.hash_id == id)
        ).first()
        if not sources:
            logger.warning(f"No item with hash_id {id} found for deletion")
            return None  # or raise an exception, or handle as needed
        session.delete(sources)
        session.commit()
        logger.info(f"Item with hash_id {id} deleted from the database")


def add_many(data: list):
    with Session(engine) as session:
        for info in data:
            # Reuse add_one function for each item
            result = add_one(info)
            if result is None:
                logger.warning(
                    f"Item with hash_id {info.get('hash_id')} could not be added"
                )
            else:
                logger.info(
                    f"Item with hash_id {info.get('hash_id')} added to the database"
                )
        session.commit()  # Commit at the end of the loop


def delete_many(ids: list):
    with Session(engine) as session:
        for id in ids:
            # Reuse delete_one function for each item
            result = delete_one(id)
            if result is None:
                logger.warning(f"No item with hash_id {id} found for deletion")
            else:
                logger.info(f"Item with hash_id {id} deleted from the database")
        session.commit()  # Commit at the end of the loop


def read_all(query: dict = None):
    with Session(engine) as session:
        statement = select(Sources)
        if query:
            statement = statement.where(
                *[getattr(Sources, key) == value for key, value in query.items()]
            )
        sources = session.exec(statement).all()
        return sources


def delete_all():
    with Session(engine) as session:
        session.exec(Sources).delete()
        session.commit()
        logger.info("All items deleted from the database")