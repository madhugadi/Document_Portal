from pydantic import BaseModel, RootModel,Field
from typing import List, Union,Dict, Any
from enum import Enum


class Metadata(BaseModel):

    Summary: List[str] = Field(default_factory=list, description="List of summary points of the document")
    Title : str
    Author : str
    DateCreated : str
    LastModified : str
    Publisher : str
    Language :str
    PageCount : Union[int, str]
    SetimentTone : str

class ChangeFormat(BaseModel):
    page :str
    changes : str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass

class PromptType(str,Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"



