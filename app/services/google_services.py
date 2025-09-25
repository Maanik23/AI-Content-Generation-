"""
Google Cloud services integration for Drive and Sheets operations.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings


class GoogleDriveService:
    """Service for Google Drive operations."""
    
    def __init__(self):
        self.credentials = self._get_credentials()
        if self.credentials:
            self.service = build('drive', 'v3', credentials=self.credentials)
        else:
            self.service = None
            logger.warning("Google Drive service disabled - no valid credentials")
    
    def _get_credentials(self):
        """Get Google Cloud credentials with enhanced error handling for n8n integration."""
        try:
            # Try environment variable first (for n8n integration)
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
                credentials_json = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_json,
                    scopes=[
                        'https://www.googleapis.com/auth/drive',
                        'https://www.googleapis.com/auth/documents',
                        'https://www.googleapis.com/auth/spreadsheets'
                    ]
                )
                logger.info("✅ Using Google credentials from environment variable (n8n integration)")
                return credentials
            
            # Try settings configuration
            if settings.google_application_credentials_json:
                credentials_json = json.loads(settings.google_application_credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_json,
                    scopes=[
                        'https://www.googleapis.com/auth/drive',
                        'https://www.googleapis.com/auth/documents',
                        'https://www.googleapis.com/auth/spreadsheets'
                    ]
                )
                logger.info("✅ Using Google credentials from settings")
                return credentials
            
            # Try file path
            if settings.google_credentials_path and os.path.exists(settings.google_credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    settings.google_credentials_path,
                    scopes=[
                        'https://www.googleapis.com/auth/drive',
                        'https://www.googleapis.com/auth/documents',
                        'https://www.googleapis.com/auth/spreadsheets'
                    ]
                )
                logger.info("✅ Using Google credentials from file")
                return credentials
            
            logger.warning("⚠️ No Google credentials found. Google Drive services will be disabled.")
            logger.info("💡 To enable Google services, set GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable in n8n")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in Google credentials: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to get Google credentials: {str(e)}")
            logger.info("💡 Please configure valid Google service account credentials")
            return None
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def download_file(self, file_id: str, local_path: str) -> str:
        """Download a file from Google Drive."""
        try:
            logger.info(f"Downloading file {file_id} to {local_path}")
            
            request = self.service.files().get_media(fileId=file_id)
            
            with open(local_path, 'wb') as f:
                f.write(request.execute())
            
            logger.info(f"Successfully downloaded file to {local_path}")
            return local_path
            
        except HttpError as e:
            logger.error(f"Google Drive API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def create_document(self, title: str, content: str, folder_id: str) -> str:
        """Create a new Google Doc in the specified folder with graceful fallback."""
        try:
            if not self.service:
                logger.warning("Google Drive service not available, creating local file instead")
                return self._create_local_fallback(title, content)
            
            logger.info(f"Creating document '{title}' in folder {folder_id}")
            
            # Create document metadata
            file_metadata = {
                'name': title,
                'parents': [folder_id],
                'mimeType': 'application/vnd.google-apps.document'
            }
            
            # Create the document
            file = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            
            # Add content to the document
            self.service.files().update(
                fileId=file_id,
                body={'name': title}
            ).execute()
            
            # Insert content
            requests = [{
                'insertText': {
                    'location': {'index': 1},
                    'text': content
                }
            }]
            
            self.service.documents().batchUpdate(
                documentId=file_id,
                body={'requests': requests}
            ).execute()
            
            logger.info(f"Successfully created document {file_id}")
            return file_id
            
        except HttpError as e:
            logger.error(f"Google Drive API error creating document: {str(e)}")
            logger.warning("Falling back to local file creation")
            return self._create_local_fallback(title, content)
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            logger.warning("Falling back to local file creation")
            return self._create_local_fallback(title, content)
    
    def _create_local_fallback(self, title: str, content: str) -> str:
        """Create a local file as fallback when Google Drive is unavailable."""
        try:
            import os
            from datetime import datetime
            
            # Create fallback directory
            fallback_dir = "docs/fallback_output"
            os.makedirs(fallback_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{title}_{timestamp}.txt"
            filepath = os.path.join(fallback_dir, filename)
            
            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Created fallback file: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to create fallback file: {str(e)}")
            return f"fallback_{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def upload_file(self, local_path: str, folder_id: str, filename: str) -> str:
        """Upload a file to Google Drive."""
        try:
            logger.info(f"Uploading file {local_path} to folder {folder_id}")
            
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            
            media = self.service.files().create(
                body=file_metadata,
                media_body=local_path,
                fields='id'
            ).execute()
            
            file_id = media.get('id')
            logger.info(f"Successfully uploaded file {file_id}")
            return file_id
            
        except HttpError as e:
            logger.error(f"Google Drive API error uploading file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def list_files_in_folder(self, folder_id: str) -> list:
        """List all files in a Google Drive folder."""
        try:
            logger.info(f"Listing files in folder {folder_id}")
            
            if not self.service:
                logger.warning("Google Drive service not available")
                return []
            
            # Query for files in the folder
            query = f"'{folder_id}' in parents and trashed=false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, size, createdTime, modifiedTime, mimeType, webViewLink)",
                orderBy="modifiedTime desc"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder {folder_id}")
            
            return files
            
        except HttpError as e:
            logger.error(f"Google Drive API error listing files: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def get_file_content(self, file_id: str) -> str:
        """Get text content from a Google Drive file."""
        try:
            logger.info(f"Getting content from file {file_id}")
            
            if not self.service:
                logger.warning("Google Drive service not available")
                return ""
            
            # Get file metadata first
            file_metadata = self.service.files().get(fileId=file_id).execute()
            mime_type = file_metadata.get('mimeType', '')
            
            # Handle different file types
            if 'google-apps.document' in mime_type:
                # Google Docs - use export
                content = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                ).execute()
                return content.decode('utf-8')
            
            elif 'google-apps.spreadsheet' in mime_type:
                # Google Sheets - use export
                content = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='text/csv'
                ).execute()
                return content.decode('utf-8')
            
            elif 'text/' in mime_type or 'application/pdf' in mime_type:
                # Text files or PDFs - download directly
                content = self.service.files().get_media(fileId=file_id).execute()
                return content.decode('utf-8')
            
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                return ""
                
        except HttpError as e:
            logger.error(f"Google Drive API error getting file content: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting file content: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def save_enhanced_content(self, content: dict, original_filename: str, folder_id: str) -> str:
        """Save enhanced content to Google Drive."""
        try:
            logger.info(f"Saving enhanced content for {original_filename}")
            
            if not self.service:
                logger.warning("Google Drive service not available")
                return ""
            
            # Create a comprehensive document with all content
            doc_title = f"Enhanced_{original_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Format content for Google Docs
            formatted_content = f"""# FIAE AI Content Factory - Enhanced Content
**Original File:** {original_filename}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Language:** German (Deutsch)

## Wissensanalyse
{content.get('knowledge_analysis', 'N/A')}

## Praktische Anwendungsfälle
{content.get('use_case_text', 'N/A')}

## Bewertungsfragen
{content.get('quiz_text', 'N/A')}

## Video-Skript
{content.get('script_text', 'N/A')}

---
*Generiert von FIAE AI Content Factory mit RAG + LangGraph + Vector Intelligence*
"""
            
            # Create the document
            file_id = self.create_document(doc_title, formatted_content, folder_id)
            
            logger.info(f"Successfully saved enhanced content: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error saving enhanced content: {str(e)}")
            raise


class GoogleSheetsService:
    """Service for Google Sheets operations."""
    
    def __init__(self):
        self.credentials = self._get_credentials()
        if self.credentials:
            self.service = build('sheets', 'v4', credentials=self.credentials)
        else:
            self.service = None
            logger.warning("Google Sheets service disabled - no valid credentials")
        self.spreadsheet_id = settings.google_sheets_id
    
    def _get_credentials(self):
        """Get Google Cloud credentials for Sheets service."""
        try:
            # Try environment variable first (for n8n integration)
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
                credentials_json = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_json,
                    scopes=[
                        'https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive'
                    ]
                )
                logger.info("✅ Using Google Sheets credentials from environment variable (n8n integration)")
                return credentials
            
            # Try settings configuration
            if settings.google_application_credentials_json:
                credentials_json = json.loads(settings.google_application_credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_json,
                    scopes=[
                        'https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive'
                    ]
                )
                logger.info("✅ Using Google Sheets credentials from settings")
                return credentials
            
            # Try file path
            if settings.google_credentials_path and os.path.exists(settings.google_credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    settings.google_credentials_path,
                    scopes=[
                        'https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive'
                    ]
                )
                logger.info("✅ Using Google Sheets credentials from file")
                return credentials
            
            logger.warning("⚠️ No Google credentials found. Google Sheets services will be disabled.")
            logger.info("💡 To enable Google Sheets, set GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable in n8n")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in Google credentials: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to get Google credentials: {str(e)}")
            logger.info("💡 Please configure valid Google service account credentials")
            return None
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def update_job_status(self, job_id: str, status: str, review_doc_links: Dict[str, str] = None) -> bool:
        """Update job status in the master Google Sheet."""
        try:
            logger.info(f"Updating job {job_id} status to {status}")
            
            # Find the row with the job_id
            range_name = 'Sheet1!A:Z'
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.warning("No data found in spreadsheet")
                return False
            
            # Find the job row
            job_row = None
            for i, row in enumerate(values):
                if len(row) > 0 and row[0] == job_id:
                    job_row = i + 1  # Sheets is 1-indexed
                    break
            
            if job_row is None:
                logger.warning(f"Job {job_id} not found in spreadsheet")
                return False
            
            # Update the status
            update_range = f'Sheet1!B{job_row}'
            update_values = [[status]]
            
            if review_doc_links:
                # Add review document links
                links_text = ", ".join([f"{k}: {v}" for k, v in review_doc_links.items()])
                update_values[0].append(links_text)
            
            body = {
                'values': update_values
            }
            
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=update_range,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Successfully updated job {job_id} status to {status}")
            return True
            
        except HttpError as e:
            logger.error(f"Google Sheets API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the current status of a job."""
        try:
            logger.info(f"Getting status for job {job_id}")
            
            range_name = 'Sheet1!A:Z'
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                return None
            
            # Find the job row
            for row in values:
                if len(row) > 0 and row[0] == job_id and len(row) > 1:
                    return row[1]  # Status is in column B
            
            return None
            
        except HttpError as e:
            logger.error(f"Google Sheets API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def add_processing_record(self, job_id: str, document_name: str, status: str, quality_score: float) -> bool:
        """Add a new processing record to the Google Sheet."""
        try:
            logger.info(f"Adding processing record for job {job_id}")
            
            if not self.service:
                logger.warning("Google Sheets service not available")
                return False
            
            # Prepare the new row data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_row = [
                job_id,
                status,
                document_name,
                timestamp,
                str(quality_score),
                "FIAE AI Content Factory",
                "RAG Enhanced Processing"
            ]
            
            # Append to the sheet
            range_name = 'Sheet1!A:G'
            body = {
                'values': [new_row]
            }
            
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Successfully added processing record for job {job_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Google Sheets API error adding record: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error adding processing record: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status from Google Sheets."""
        try:
            logger.info("Getting processing status from Google Sheets")
            
            if not self.service:
                logger.warning("Google Sheets service not available")
                return {"last_processing": None, "total_processed": 0}
            
            range_name = 'Sheet1!A:G'
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                return {"last_processing": None, "total_processed": 0}
            
            # Calculate statistics
            total_processed = len(values) - 1  # Subtract header row
            completed_count = sum(1 for row in values[1:] if len(row) > 1 and row[1] == "completed")
            failed_count = sum(1 for row in values[1:] if len(row) > 1 and row[1] == "failed")
            
            # Get last processing time
            last_processing = None
            if len(values) > 1:
                last_row = values[-1]
                if len(last_row) > 3:
                    last_processing = last_row[3]  # Timestamp column
            
            return {
                "last_processing": last_processing,
                "total_processed": total_processed,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "processing_rate": f"{(completed_count/total_processed*100):.1f}%" if total_processed > 0 else "0%"
            }
            
        except HttpError as e:
            logger.error(f"Google Sheets API error getting status: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting processing status: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending HITL approval requests."""
        try:
            logger.info("Getting pending approvals from Google Sheets")
            
            if not self.service:
                logger.warning("Google Sheets service not available")
                return []
            
            range_name = 'Sheet1!A:G'
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                return []
            
            # Find pending approvals
            pending_approvals = []
            for i, row in enumerate(values[1:], start=2):  # Skip header, start from row 2
                if len(row) > 1 and row[1] == "awaiting_script_approval":
                    pending_approvals.append({
                        "approval_id": f"approval_{row[0]}_{i}",
                        "job_id": row[0],
                        "document_name": row[2] if len(row) > 2 else "Unknown",
                        "status": row[1],
                        "timestamp": row[3] if len(row) > 3 else None,
                        "row_number": i
                    })
            
            logger.info(f"Found {len(pending_approvals)} pending approvals")
            return pending_approvals
            
        except HttpError as e:
            logger.error(f"Google Sheets API error getting approvals: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting pending approvals: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def get_approval_request(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific approval request."""
        try:
            logger.info(f"Getting approval request {approval_id}")
            
            pending_approvals = self.get_pending_approvals()
            for approval in pending_approvals:
                if approval["approval_id"] == approval_id:
                    return approval
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting approval request: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def update_approval_status(self, approval_id: str, status: str, approved_by: str = "", notes: str = "", revision_requests: List[str] = None) -> bool:
        """Update approval status in Google Sheets."""
        try:
            logger.info(f"Updating approval {approval_id} status to {status}")
            
            if not self.service:
                logger.warning("Google Sheets service not available")
                return False
            
            # Find the approval request
            approval = self.get_approval_request(approval_id)
            if not approval:
                logger.warning(f"Approval request {approval_id} not found")
                return False
            
            # Update the status
            row_number = approval["row_number"]
            update_range = f'Sheet1!B{row_number}'
            update_values = [[status]]
            
            # Add approval details if provided
            if approved_by or notes:
                details = f"Approved by: {approved_by}" if approved_by else ""
                if notes:
                    details += f" | Notes: {notes}"
                if revision_requests:
                    details += f" | Revisions: {', '.join(revision_requests)}"
                update_values[0].append(details)
            
            body = {
                'values': update_values
            }
            
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=update_range,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Successfully updated approval {approval_id} status to {status}")
            return True
            
        except HttpError as e:
            logger.error(f"Google Sheets API error updating approval: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error updating approval status: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get HITL approval statistics."""
        try:
            logger.info("Getting approval statistics from Google Sheets")
            
            if not self.service:
                logger.warning("Google Sheets service not available")
                return {"total_approvals": 0, "approved": 0, "rejected": 0, "pending": 0}
            
            range_name = 'Sheet1!A:G'
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                return {"total_approvals": 0, "approved": 0, "rejected": 0, "pending": 0}
            
            # Calculate statistics
            total_approvals = len(values) - 1  # Subtract header row
            approved = sum(1 for row in values[1:] if len(row) > 1 and "approved" in row[1].lower())
            rejected = sum(1 for row in values[1:] if len(row) > 1 and "rejected" in row[1].lower())
            pending = sum(1 for row in values[1:] if len(row) > 1 and "awaiting" in row[1].lower())
            
            return {
                "total_approvals": total_approvals,
                "approved": approved,
                "rejected": rejected,
                "pending": pending,
                "approval_rate": f"{(approved/(approved+rejected)*100):.1f}%" if (approved+rejected) > 0 else "0%"
            }
            
        except HttpError as e:
            logger.error(f"Google Sheets API error getting statistics: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting approval statistics: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, exp_base=settings.retry_backoff_factor),
        retry=retry_if_exception_type((HttpError, ConnectionError, TimeoutError))
    )
    def update_job_status(self, job_id: str, status: str, rejection_reason: str = "", regenerate_phase: str = "") -> bool:
        """Update job status with additional details."""
        try:
            logger.info(f"Updating job {job_id} status to {status}")
            
            if not self.service:
                logger.warning("Google Sheets service not available")
                return False
            
            # Find the job row
            range_name = 'Sheet1!A:Z'
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.warning("No data found in spreadsheet")
                return False
            
            # Find the job row
            job_row = None
            for i, row in enumerate(values):
                if len(row) > 0 and row[0] == job_id:
                    job_row = i + 1  # Sheets is 1-indexed
                    break
            
            if job_row is None:
                logger.warning(f"Job {job_id} not found in spreadsheet")
                return False
            
            # Update the status with additional details
            update_range = f'Sheet1!B{job_row}'
            update_values = [[status]]
            
            # Add additional details
            details = []
            if rejection_reason:
                details.append(f"Rejection: {rejection_reason}")
            if regenerate_phase:
                details.append(f"Regenerate: {regenerate_phase}")
            
            if details:
                update_values[0].append(" | ".join(details))
            
            body = {
                'values': update_values
            }
            
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=update_range,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Successfully updated job {job_id} status to {status}")
            return True
            
        except HttpError as e:
            logger.error(f"Google Sheets API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
            raise
