from typing import Optional
from pydantic import BaseModel


class ManualInfo(BaseModel):
    unique_id: str
    metadata: dict
    page_content: Optional[str]