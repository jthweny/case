import re
import mailbox
import email
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from datetime import datetime
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2 not available. Install with: pip install PyPDF2")
import json
import hashlib
import os

class DocumentParser:
    """
    Parses various document types into individual messages/items
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_timeline_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse the Nadia timeline format file into individual messages"""
        messages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Updated pattern to match the actual format:
            # Format: "   1. [timestamp] [message_type] sender_name\n      Content: message_content"
            pattern = r'\s*(\d+)\.\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+([^\n]+)\n\s+Content:\s+(.+?)(?=\n\s*\d+\.|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for match in matches:
                entry_num, timestamp, message_type, sender, content_text = match
                
                # Clean up content
                content_text = content_text.strip()
                if content_text == "[Message content not available]":
                    content_text = ""
                
                message = {
                    'entry_number': int(entry_num),
                    'timestamp': timestamp,
                    'message_type': message_type,
                    'sender': sender,
                    'content': content_text,
                    'source_file': file_path,
                    'document_type': 'timeline_message'
                }
                messages.append(message)
            
            self.logger.info(f"Parsed {len(messages)} messages from timeline file")
            
            # If no messages found with the pattern, try to extract as plain text
            if not messages:
                self.logger.warning("No timeline entries found with pattern, treating as plain text")
                # Return the entire file as a single document
                return [{
                    'entry_number': 1,
                    'timestamp': '',
                    'message_type': 'TIMELINE_FILE',
                    'sender': 'System',
                    'content': content,
                    'source_file': file_path,
                    'document_type': 'timeline_document'
                }]
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error parsing timeline file: {e}")
            # Fallback: return as plain text document
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return [{
                    'entry_number': 1,
                    'timestamp': '',
                    'message_type': 'TIMELINE_FILE',
                    'sender': 'System',
                    'content': content,
                    'source_file': file_path,
                    'document_type': 'timeline_document'
                }]
            except:
                return []
    
    def parse_mbox_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse .mbox email files into individual emails"""
        emails = []
        
        try:
            mbox = mailbox.mbox(file_path)
            
            for i, message in enumerate(mbox):
                try:
                    # Extract email details
                    subject = message.get('Subject', 'No Subject')
                    sender = message.get('From', 'Unknown Sender')
                    recipient = message.get('To', 'Unknown Recipient')
                    date = message.get('Date', '')
                    
                    # Get email body
                    body = ""
                    if message.is_multipart():
                        for part in message.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                break
                    else:
                        body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                    
                    email_data = {
                        'entry_number': i + 1,
                        'timestamp': date,
                        'message_type': 'EMAIL',
                        'sender': sender,
                        'recipient': recipient,
                        'subject': subject,
                        'content': body,
                        'source_file': file_path,
                        'document_type': 'email'
                    }
                    emails.append(email_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing email {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(emails)} emails from mbox file")
            return emails
            
        except Exception as e:
            self.logger.error(f"Error parsing mbox file: {e}")
            return []
    
    def parse_pdf_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse PDF file and return as a single document"""
        if not PDF_AVAILABLE:
            self.logger.warning("PyPDF2 not available, skipping PDF parsing")
            return None
            
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += page.extract_text() + "\n"
                
                pdf_data = {
                    'entry_number': 1,
                    'timestamp': '',
                    'message_type': 'DOCUMENT',
                    'sender': 'System',
                    'content': text_content,
                    'source_file': file_path,
                    'document_type': 'pdf',
                    'page_count': len(pdf_reader.pages)
                }
                
                self.logger.info(f"Parsed PDF with {len(pdf_reader.pages)} pages")
                return pdf_data
                
        except Exception as e:
            self.logger.error(f"Error parsing PDF file: {e}")
            return None
    
    def chunk_large_content(self, content: str, max_chars: int = 8000) -> List[str]:
        """Break large content into chunks for processing"""
        if len(content) <= max_chars:
            return [content]
        
        chunks = []
        current_chunk = ""
        sentences = content.split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        self.logger.info(f"Split content into {len(chunks)} chunks")
        return chunks
    
    def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse any supported document type"""
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        
        if extension == '.txt' and 'timeline' in path_obj.name.lower():
            return self.parse_timeline_file(file_path)
        elif extension == '.mbox':
            return self.parse_mbox_file(file_path)
        elif extension == '.pdf':
            pdf_data = self.parse_pdf_file(file_path)
            return [pdf_data] if pdf_data else []
        else:
            # Try to parse as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return [{
                    'entry_number': 1,
                    'timestamp': '',
                    'message_type': 'TEXT_FILE',
                    'sender': 'System',
                    'content': content,
                    'source_file': file_path,
                    'document_type': 'text_file'
                }]
            except Exception as e:
                self.logger.error(f"Unable to parse file {file_path}: {e}")
                return []
    
    def get_all_documents(self, case_docs_dir: str) -> List[Dict[str, Any]]:
        """Get all documents from the case docs directory"""
        all_documents = []
        case_path = Path(case_docs_dir)
        
        if not case_path.exists():
            self.logger.error(f"Case docs directory not found: {case_docs_dir}")
            return []
        
        # Recursively find all relevant files
        for file_path in case_path.rglob('*'):
            if file_path.is_file():
                extension = file_path.suffix.lower()
                if extension in ['.txt', '.mbox', '.pdf'] or 'timeline' in file_path.name.lower():
                    documents = self.parse_document(str(file_path))
                    all_documents.extend(documents)
        
        self.logger.info(f"Found {len(all_documents)} total documents")
        return all_documents
