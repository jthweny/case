"""
Database manager for legal case analysis system.
Handles storage and retrieval of message analyses, chunks, and findings.
"""

import aiosqlite
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from loguru import logger


class DatabaseManager:
    """Manages all database operations for the legal case analysis system."""
    
    def __init__(self, db_path: str = "data/analysis.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def initialize(self):
        """Initialize database connection and create tables if needed."""
        self._connection = await aiosqlite.connect(str(self.db_path))
        self._connection.row_factory = aiosqlite.Row
        
        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")
        
        # Create tables
        await self._create_tables()
        await self._connection.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    async def _create_tables(self):
        """Create all necessary tables for the analysis system."""
        
        # Documents catalog table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER,
                created_date DATETIME,
                processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Message analysis table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS message_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                message_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME,
                sender TEXT,
                message_type TEXT,
                content TEXT,
                
                -- Analysis results
                is_relevant BOOLEAN DEFAULT 0,
                relevance_score REAL DEFAULT 0.0,
                brief_summary TEXT,
                
                -- Key extractions
                mentions_property BOOLEAN DEFAULT 0,
                mentions_tax BOOLEAN DEFAULT 0,
                mentions_domicile BOOLEAN DEFAULT 0,
                mentions_timeline BOOLEAN DEFAULT 0,
                
                -- Legal research
                needs_legal_research BOOLEAN DEFAULT 0,
                legal_references TEXT,
                jurisdiction TEXT,
                
                -- Tags and connections
                issue_tags TEXT,
                contradictions TEXT,
                
                -- Processing metadata
                processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                tokens_used INTEGER,
                processing_error TEXT,
                
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Create indexes for faster queries
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_relevance 
            ON message_analysis(is_relevant, relevance_score DESC)
        """)
        
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_timestamp 
            ON message_analysis(timestamp)
        """)
        
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_sender 
            ON message_analysis(sender)
        """)
        
        # Chunks table for vector storage
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                embedding BLOB,
                
                -- Relevance scoring
                baseline_score REAL DEFAULT 0.0,
                confidence_score REAL DEFAULT 0.0,
                issue_labels TEXT,
                
                -- Metadata
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Legal research findings table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS legal_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                research_query TEXT,
                jurisdiction TEXT,
                findings TEXT,
                sources TEXT,
                research_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (message_id) REFERENCES message_analysis(message_id)
            )
        """)
        
        # Processing log table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time DATETIME,
                end_time DATETIME,
                items_processed INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
    
    async def add_document(self, file_path: str, file_type: str, 
                          file_size: int = None, metadata: Dict = None) -> int:
        """Add a document to the catalog."""
        try:
            cursor = await self._connection.execute("""
                INSERT OR IGNORE INTO documents (file_path, file_type, file_size, metadata)
                VALUES (?, ?, ?, ?)
            """, (file_path, file_type, file_size, json.dumps(metadata) if metadata else None))
            
            await self._connection.commit()
            
            # Get the document ID
            if cursor.lastrowid:
                return cursor.lastrowid
            else:
                # Document already exists, get its ID
                cursor = await self._connection.execute(
                    "SELECT id FROM documents WHERE file_path = ?", (file_path,)
                )
                row = await cursor.fetchone()
                return row["id"] if row else None
                
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            raise
    
    async def save_message_analysis(self, analysis_data: Dict[str, Any]) -> bool:
        """Save message analysis results to database."""
        try:
            # Convert lists/dicts to JSON strings
            if "issue_tags" in analysis_data and isinstance(analysis_data["issue_tags"], list):
                analysis_data["issue_tags"] = json.dumps(analysis_data["issue_tags"])
            
            if "legal_references" in analysis_data and isinstance(analysis_data["legal_references"], list):
                analysis_data["legal_references"] = json.dumps(analysis_data["legal_references"])
            
            if "contradictions" in analysis_data and isinstance(analysis_data["contradictions"], list):
                analysis_data["contradictions"] = json.dumps(analysis_data["contradictions"])
            
            # Build the INSERT query dynamically based on provided fields
            fields = list(analysis_data.keys())
            placeholders = ["?" for _ in fields]
            values = [analysis_data[field] for field in fields]
            
            query = f"""
                INSERT OR REPLACE INTO message_analysis ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
            """
            
            await self._connection.execute(query, values)
            await self._connection.commit()
            
            logger.debug(f"Saved analysis for message {analysis_data.get('message_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving message analysis: {e}")
            return False
    
    async def get_relevant_messages(self, min_score: float = 0.5) -> List[Dict]:
        """Retrieve all relevant messages above a certain score threshold."""
        cursor = await self._connection.execute("""
            SELECT * FROM message_analysis
            WHERE is_relevant = 1 AND relevance_score >= ?
            ORDER BY relevance_score DESC, timestamp
        """, (min_score,))
        
        rows = await cursor.fetchall()
        
        # Convert rows to dicts and parse JSON fields
        messages = []
        for row in rows:
            msg = dict(row)
            
            # Parse JSON fields
            for field in ["issue_tags", "legal_references", "contradictions"]:
                if msg.get(field):
                    try:
                        msg[field] = json.loads(msg[field])
                    except:
                        pass
            
            messages.append(msg)
        
        return messages
    
    async def get_messages_by_tags(self, tags: List[str]) -> List[Dict]:
        """Retrieve messages that have any of the specified tags."""
        # Build a query that checks if any tag is in the JSON array
        tag_conditions = []
        params = []
        
        for tag in tags:
            tag_conditions.append("issue_tags LIKE ?")
            params.append(f'%"{tag}"%')
        
        query = f"""
            SELECT * FROM message_analysis
            WHERE is_relevant = 1 AND ({' OR '.join(tag_conditions)})
            ORDER BY timestamp
        """
        
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    async def get_messages_by_sender(self, sender: str) -> List[Dict]:
        """Retrieve all messages from a specific sender."""
        cursor = await self._connection.execute("""
            SELECT * FROM message_analysis
            WHERE sender = ?
            ORDER BY timestamp
        """, (sender,))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def get_messages_needing_research(self) -> List[Dict]:
        """Get messages that need legal research but haven't been researched yet."""
        cursor = await self._connection.execute("""
            SELECT ma.* FROM message_analysis ma
            LEFT JOIN legal_research lr ON ma.message_id = lr.message_id
            WHERE ma.needs_legal_research = 1 AND lr.id IS NULL
            ORDER BY ma.relevance_score DESC
        """)
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def save_legal_research(self, message_id: str, research_data: Dict):
        """Save legal research findings for a message."""
        await self._connection.execute("""
            INSERT INTO legal_research (message_id, research_query, jurisdiction, findings, sources)
            VALUES (?, ?, ?, ?, ?)
        """, (
            message_id,
            research_data.get("query"),
            research_data.get("jurisdiction"),
            json.dumps(research_data.get("findings", [])),
            json.dumps(research_data.get("sources", []))
        ))
        
        await self._connection.commit()
    
    async def log_processing_phase(self, phase: str, status: str, 
                                  start_time: datetime = None, 
                                  end_time: datetime = None,
                                  items_processed: int = 0,
                                  errors: int = 0,
                                  metadata: Dict = None):
        """Log processing phase progress."""
        await self._connection.execute("""
            INSERT INTO processing_log 
            (phase, status, start_time, end_time, items_processed, errors, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            phase, status, start_time, end_time, 
            items_processed, errors,
            json.dumps(metadata) if metadata else None
        ))
        
        await self._connection.commit()
    
    async def get_processing_stats(self) -> Dict:
        """Get overall processing statistics."""
        stats = {}
        
        # Total documents
        cursor = await self._connection.execute("SELECT COUNT(*) as count FROM documents")
        stats["total_documents"] = (await cursor.fetchone())["count"]
        
        # Total messages analyzed
        cursor = await self._connection.execute("SELECT COUNT(*) as count FROM message_analysis")
        stats["total_messages"] = (await cursor.fetchone())["count"]
        
        # Relevant messages
        cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM message_analysis WHERE is_relevant = 1"
        )
        stats["relevant_messages"] = (await cursor.fetchone())["count"]
        
        # Messages by tag
        cursor = await self._connection.execute("""
            SELECT issue_tags, COUNT(*) as count 
            FROM message_analysis 
            WHERE issue_tags IS NOT NULL 
            GROUP BY issue_tags
        """)
        
        tag_counts = {}
        for row in await cursor.fetchall():
            try:
                tags = json.loads(row["issue_tags"])
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except:
                pass
        
        stats["tags"] = tag_counts
        
        # Processing phases
        cursor = await self._connection.execute("""
            SELECT phase, status, items_processed, errors
            FROM processing_log
            ORDER BY id DESC
        """)
        
        stats["processing_phases"] = [dict(row) for row in await cursor.fetchall()]
        
        return stats
    
    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            logger.info("Database connection closed")


# Example usage and testing
async def test_database():
    """Test database functionality."""
    db = DatabaseManager()
    await db.initialize()
    
    # Test adding a document
    doc_id = await db.add_document(
        file_path="/path/to/test.pdf",
        file_type="pdf",
        file_size=1024,
        metadata={"source": "email"}
    )
    
    print(f"Added document with ID: {doc_id}")
    
    # Test saving message analysis
    await db.save_message_analysis({
        "document_id": doc_id,
        "message_id": "msg_001",
        "timestamp": datetime.now(),
        "sender": "Nadia",
        "message_type": "email",
        "content": "Test message about Gibraltar property",
        "is_relevant": True,
        "relevance_score": 0.95,
        "brief_summary": "Discusses Gibraltar property ownership",
        "mentions_property": True,
        "mentions_tax": True,
        "mentions_domicile": False,
        "needs_legal_research": True,
        "jurisdiction": "Gibraltar",
        "issue_tags": ["property", "tax", "gibraltar"],
        "model_used": "gemini-2.5-pro",
        "tokens_used": 500
    })
    
    # Test retrieving messages
    relevant_msgs = await db.get_relevant_messages(min_score=0.8)
    print(f"Found {len(relevant_msgs)} relevant messages")
    
    # Get stats
    stats = await db.get_processing_stats()
    print(f"Processing stats: {stats}")
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(test_database())