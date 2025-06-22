"""
Message processor for analyzing individual messages/emails with Gemini.
Determines relevance, creates summaries, and triggers legal research.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import mailbox
import email
from email.utils import parsedate_to_datetime

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .smart_api_manager import SmartAPIManager
from .hybrid_database import HybridDatabaseManager

console = Console()


class MessageProcessor:
    """Processes individual messages for legal case analysis."""
    
    def __init__(self, db_manager: DatabaseManager, api_manager: GeminiAPIManager):
        self.db = db_manager
        self.api = api_manager
        
        # Analysis prompt template
        self.analysis_prompt_template = """
Analyze this message for relevance to a probate/inheritance legal case involving:
- Deceased: Sean Thweny
- Key party: Nadia (claims Gibraltar residency, administered estate after 6-year delay)
- Issues: UK domicile claims, Gibraltar/Portugal property transfers, tax implications

Message details:
Sender: {sender}
Date: {date}
Type: {msg_type}
Content: {content}

Provide your analysis in the following JSON format:
{{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "brief_summary": "2-3 sentence summary if relevant, or null if not",
    "mentions_property": true/false,
    "mentions_tax": true/false,
    "mentions_domicile": true/false,
    "mentions_timeline": true/false,
    "needs_legal_research": true/false,
    "legal_topic": "specific legal topic if research needed",
    "jurisdiction": "Gibraltar/UK/Portugal/null",
    "issue_tags": ["tag1", "tag2"],
    "key_facts": ["fact1", "fact2"]
}}

Focus on: property ownership, tax residency, domicile claims, timeline inconsistencies, 
administrative delays, and any contradictions in statements.
"""
        
        # System instruction for quality analysis
        self.system_instruction = """You are an expert legal analyst specializing in international 
probate and tax law. Analyze messages for relevance to inheritance disputes, focusing on 
jurisdictional issues between UK, Gibraltar, and Portugal. Be precise and identify subtle 
implications that could affect tax liability or property rights."""
    
    async def process_mbox_file(self, mbox_path: str, doc_id: int) -> Dict[str, Any]:
        """Process all messages in an mbox file."""
        stats = {
            "total_messages": 0,
            "relevant_messages": 0,
            "errors": 0,
            "needs_research": 0
        }
        
        mbox_path = Path(mbox_path)
        if not mbox_path.exists():
            logger.error(f"Mbox file not found: {mbox_path}")
            return stats
        
        # Open mbox file
        mbox = mailbox.mbox(str(mbox_path))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Processing {mbox_path.name}...", 
                total=len(mbox)
            )
            
            for idx, message in enumerate(mbox):
                try:
                    # Extract message data
                    msg_data = self._extract_email_data(message, idx)
                    msg_data["document_id"] = doc_id
                    
                    # Analyze message
                    analysis = await self.analyze_message(msg_data)
                    
                    if analysis:
                        # Save to database
                        analysis_data = {
                            **msg_data,
                            **analysis
                        }
                        await self.db.save_message_analysis(analysis_data)
                        
                        stats["total_messages"] += 1
                        if analysis.get("is_relevant"):
                            stats["relevant_messages"] += 1
                        if analysis.get("needs_legal_research"):
                            stats["needs_research"] += 1
                    else:
                        stats["errors"] += 1
                    
                    progress.update(task, advance=1)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing message {idx}: {e}")
                    stats["errors"] += 1
                    progress.update(task, advance=1)
        
        return stats
    
    async def process_text_timeline(self, timeline_path: str, doc_id: int) -> Dict[str, Any]:
        """Process timestamped text/audio timeline file."""
        stats = {
            "total_messages": 0,
            "relevant_messages": 0,
            "errors": 0,
            "needs_research": 0
        }
        
        timeline_path = Path(timeline_path)
        if not timeline_path.exists():
            logger.error(f"Timeline file not found: {timeline_path}")
            return stats
        
        # Read timeline file
        with open(timeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse timeline entries (adjust regex based on actual format)
        # Example: [2019-04-10 10:33:36.637000] [TEXT_MESSAGE] Nadia Thweny
        #          Content: message text here
        pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\] \[([^\]]+)\] ([^\n]+)\s*\n\s*(?:Content: )?(.*?)(?=\n\s*\n|\Z)'
        
        entries = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Processing timeline entries...", 
                total=len(entries)
            )
            
            for idx, (timestamp, msg_type, sender, content) in enumerate(entries):
                try:
                    # Create message data
                    msg_data = {
                        "document_id": doc_id,
                        "message_id": f"timeline_{idx}_{timestamp.replace(' ', '_')}",
                        "timestamp": datetime.fromisoformat(timestamp.split('.')[0]),
                        "sender": sender.strip(),
                        "message_type": msg_type.lower().replace('_', ' '),
                        "content": content.strip()
                    }
                    
                    # Skip empty messages
                    if not msg_data["content"]:
                        progress.update(task, advance=1)
                        continue
                    
                    # Analyze message
                    analysis = await self.analyze_message(msg_data)
                    
                    if analysis:
                        # Save to database
                        analysis_data = {
                            **msg_data,
                            **analysis
                        }
                        await self.db.save_message_analysis(analysis_data)
                        
                        stats["total_messages"] += 1
                        if analysis.get("is_relevant"):
                            stats["relevant_messages"] += 1
                        if analysis.get("needs_legal_research"):
                            stats["needs_research"] += 1
                    else:
                        stats["errors"] += 1
                    
                    progress.update(task, advance=1)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing timeline entry {idx}: {e}")
                    stats["errors"] += 1
                    progress.update(task, advance=1)
        
        return stats
    
    def _extract_email_data(self, message: email.message.Message, idx: int) -> Dict[str, Any]:
        """Extract relevant data from an email message."""
        # Get sender
        sender = message.get('From', 'Unknown')
        # Extract just the name or email
        if '<' in sender:
            sender = sender.split('<')[0].strip().strip('"')
        
        # Get date
        date_str = message.get('Date', '')
        try:
            msg_date = parsedate_to_datetime(date_str) if date_str else datetime.now()
        except:
            msg_date = datetime.now()
        
        # Get subject
        subject = message.get('Subject', 'No Subject')
        
        # Get body
        body = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        pass
        else:
            try:
                body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(message.get_payload())
        
        # Combine subject and body for analysis
        content = f"Subject: {subject}\n\n{body}".strip()
        
        return {
            "message_id": f"email_{idx}_{msg_date.isoformat()}",
            "timestamp": msg_date,
            "sender": sender,
            "message_type": "email",
            "content": content[:5000]  # Limit content length
        }
    
    async def analyze_message(self, msg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single message using Gemini API."""
        try:
            # Format the prompt
            prompt = self.analysis_prompt_template.format(
                sender=msg_data.get("sender", "Unknown"),
                date=msg_data.get("timestamp", "Unknown"),
                msg_type=msg_data.get("message_type", "Unknown"),
                content=msg_data.get("content", "")
            )
            
            # Call Gemini API
            response = await self.api.generate_content(
                prompt=prompt,
                system_instruction=self.system_instruction,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1000
            )
            
            if not response:
                logger.error(f"No response from API for message {msg_data.get('message_id')}")
                return None
            
            # Parse JSON response
            try:
                # Clean the response (remove markdown code blocks if present)
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                
                analysis = json.loads(cleaned_response.strip())
                
                # Add metadata
                analysis["model_used"] = self.api.generation_model
                analysis["tokens_used"] = len(prompt.split()) + len(response.split())  # Rough estimate
                
                # Log relevant messages
                if analysis.get("is_relevant"):
                    logger.info(
                        f"Found relevant message from {msg_data.get('sender')}: "
                        f"{analysis.get('brief_summary', '')[:100]}..."
                    )
                
                return analysis
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response was: {response}")
                
                # Try to extract basic relevance
                return {
                    "is_relevant": "relevant" in response.lower(),
                    "relevance_score": 0.5,
                    "brief_summary": "Failed to parse detailed analysis",
                    "processing_error": f"JSON parse error: {str(e)}",
                    "model_used": self.api.generation_model
                }
                
        except Exception as e:
            logger.error(f"Error analyzing message {msg_data.get('message_id')}: {e}")
            return None
    
    async def process_all_documents(self, docs_path: str):
        """Process all documents in the case docs directory."""
        docs_path = Path(docs_path)
        
        # Track overall statistics
        total_stats = {
            "documents_processed": 0,
            "total_messages": 0,
            "relevant_messages": 0,
            "needs_research": 0,
            "errors": 0
        }
        
        # Log processing start
        await self.db.log_processing_phase(
            phase="message_analysis",
            status="started",
            start_time=datetime.now()
        )
        
        # Process each file type
        for file_path in docs_path.glob("**/*"):
            if file_path.is_file():
                try:
                    # Add document to catalog
                    doc_id = await self.db.add_document(
                        file_path=str(file_path),
                        file_type=file_path.suffix.lower(),
                        file_size=file_path.stat().st_size
                    )
                    
                    stats = None
                    
                    # Process based on file type
                    if file_path.suffix.lower() == ".mbox":
                        console.print(f"\n[bold blue]Processing email archive: {file_path.name}[/bold blue]")
                        stats = await self.process_mbox_file(str(file_path), doc_id)
                    
                    elif "timeline" in file_path.name.lower() and file_path.suffix.lower() == ".txt":
                        console.print(f"\n[bold blue]Processing timeline: {file_path.name}[/bold blue]")
                        stats = await self.process_text_timeline(str(file_path), doc_id)
                    
                    if stats:
                        total_stats["documents_processed"] += 1
                        total_stats["total_messages"] += stats["total_messages"]
                        total_stats["relevant_messages"] += stats["relevant_messages"]
                        total_stats["needs_research"] += stats["needs_research"]
                        total_stats["errors"] += stats["errors"]
                        
                        console.print(f"[green]✓ Processed {stats['total_messages']} messages, "
                                    f"{stats['relevant_messages']} relevant[/green]")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    total_stats["errors"] += 1
        
        # Log processing completion
        await self.db.log_processing_phase(
            phase="message_analysis",
            status="completed",
            end_time=datetime.now(),
            items_processed=total_stats["total_messages"],
            errors=total_stats["errors"],
            metadata=total_stats
        )
        
        # Print summary
        console.print("\n[bold green]Processing Complete![/bold green]")
        console.print(f"Documents processed: {total_stats['documents_processed']}")
        console.print(f"Total messages: {total_stats['total_messages']}")
        console.print(f"Relevant messages: {total_stats['relevant_messages']}")
        console.print(f"Need legal research: {total_stats['needs_research']}")
        console.print(f"Errors: {total_stats['errors']}")
        
        return total_stats


# Main execution
async def main():
    """Main function to run message processing."""
    # Initialize managers
    db = DatabaseManager()
    await db.initialize()
    
    api = GeminiAPIManager()
    
    # Create processor
    processor = MessageProcessor(db, api)
    
    # Process all documents
    docs_path = os.getenv("CASE_DOCS_PATH", "/home/joshuathweny/Desktop/case docs ONLY/")
    await processor.process_all_documents(docs_path)
    
    # Close database
    await db.close()


if __name__ == "__main__":
    import os
    asyncio.run(main())