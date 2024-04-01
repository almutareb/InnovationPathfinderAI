from sqlmodel import SQLModel, Field
from typing import Optional

import datetime

class Sources(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field()
    title: Optional[str] = Field(default="NA", unique=False)
    hash_id: str = Field(unique=True)
    created_at: float = Field(default=datetime.datetime.now().timestamp())
    summary: str = Field(default="")
    embedded: bool = Field(default=False)

    __table_args__ = {"extend_existing": True}