"""
Pydantic models for request/response schemas and data validation.
"""

from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict, Any, Optional
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status enumeration for job processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    REVIEW = "review"
    APPROVED = "approved"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessDocumentRequest(BaseModel):
    """Request model for document processing endpoint."""
    file_path: Optional[str] = Field(None, description="Path to the source .docx file")
    content: Optional[str] = Field(None, description="Direct document content (for n8n workflow)")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if v and not v.endswith('.docx'):
            raise ValueError('File must be a .docx file')
        return v
    
    @model_validator(mode='after')
    def validate_content_or_file_path(self):
        if not self.file_path and not self.content:
            raise ValueError('Either file_path or content must be provided')
        return self


class ProcessDocumentResponse(BaseModel):
    """Response model for document processing endpoint."""
    success: bool = Field(..., description="Whether the processing was successful")
    job_id: str = Field(..., description="Unique job identifier")
    message: str = Field(..., description="Status message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Generated content data")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class GeneratedContent(BaseModel):
    """Model for generated content from all agents."""
    use_case_text: str = Field(..., description="Generated practical use cases in German")
    quiz_text: str = Field(..., description="Generated quiz content in German")
    script_text: str = Field(..., description="Generated video script in German")
    audio_script: str = Field(..., description="Script text for audio generation in German")
    
    # Metadata
    source_document: str = Field(..., description="Source document path")
    processing_timestamp: str = Field(..., description="Processing timestamp")
    job_id: str = Field(..., description="Job identifier")
    language: str = Field(default="de", description="Content language (German)")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Check timestamp")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")
