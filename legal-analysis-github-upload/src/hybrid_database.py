import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")
import hashlib
import uuid

class HybridDatabaseManager:
    """
    Hybrid database manager using SQLite for metadata and ChromaDB for semantic search
    Enhanced for Phase 2 AI Analysis: analyze each document with Gemini, use baseline doc, trigger Firecrawl, store insights.
    """
    def __init__(self, db_path: str = "data/analysis.db", chroma_path: str = "data/chroma_db"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize SQLite
        self._init_sqlite()
        
        # Ensure schema is up to date
        self._ensure_schema_updated()
        
        # Initialize ChromaDB
        self._init_chromadb()
        
    def _ensure_schema_updated(self):
        """Ensure database schema is up to date with all required columns"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if processing_status column exists in documents table
            cursor.execute("PRAGMA table_info(documents)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'processing_status' not in columns:
                self.logger.info("Adding missing processing_status column to documents table")
                cursor.execute("ALTER TABLE documents ADD COLUMN processing_status TEXT DEFAULT 'pending'")
            
            if 'confidence_score' not in columns:
                self.logger.info("Adding missing confidence_score column to documents table")
                cursor.execute("ALTER TABLE documents ADD COLUMN confidence_score REAL DEFAULT 0.0")
            
            if 'created_at' not in columns:
                self.logger.info("Adding missing created_at column to documents table")
                cursor.execute("ALTER TABLE documents ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
            
            if 'updated_at' not in columns:
                self.logger.info("Adding missing updated_at column to documents table")
                cursor.execute("ALTER TABLE documents ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
            
            conn.commit()

    def _init_sqlite(self):
        """Initialize SQLite database with enhanced schema"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    timestamp TEXT,
                    sender TEXT,
                    content_hash TEXT,
                    content_preview TEXT,
                    character_count INTEGER,
                    confidence_score REAL DEFAULT 0.0,
                    processing_status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    prompt_used TEXT,
                    ai_response TEXT,
                    confidence_score REAL,
                    relevance_score REAL,
                    legal_facts TEXT,
                    legal_laws TEXT,
                    firecrawl_triggered BOOLEAN DEFAULT FALSE,
                    firecrawl_results TEXT,
                    processing_time REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Research sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    session_name TEXT,
                    prompt_template TEXT,
                    settings TEXT,
                    progress_data TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Deep research results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deep_research (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    research_query TEXT,
                    related_documents TEXT,
                    findings TEXT,
                    contradictions TEXT,
                    confidence_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES research_sessions (id)
                )
            """)
            
            # UI state persistence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ui_state (
                    id TEXT PRIMARY KEY DEFAULT 'current',
                    last_prompt TEXT,
                    last_output TEXT,
                    current_settings TEXT,
                    session_history TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
    def _init_chromadb(self):
        """Initialize ChromaDB for semantic search"""
        if not CHROMADB_AVAILABLE:
            self.logger.warning("ChromaDB not available, semantic search disabled")
            return
            
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collections
            try:
                self.documents_collection = self.chroma_client.get_collection("documents")
            except:
                self.documents_collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"description": "Document content for semantic search"}
                )
                
            try:
                self.analysis_collection = self.chroma_client.get_collection("analysis")
            except:
                self.analysis_collection = self.chroma_client.create_collection(
                    name="analysis",
                    metadata={"description": "Analysis results for semantic search"}
                )
        except Exception as e:
            self.logger.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_client = None
    
    def store_document(self, source_file: str, document_type: str, content: str, 
                      timestamp: Optional[str] = None, sender: Optional[str] = None, 
                      confidence_score: float = 0.0) -> str:
        """Store a document with both SQLite metadata and ChromaDB content"""
        
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        content_preview = content[:200] + "..." if len(content) > 200 else content
        
        # Store in SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents 
                (id, source_file, document_type, timestamp, sender, content_hash, 
                 content_preview, character_count, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, source_file, document_type, timestamp, sender, 
                  content_hash, content_preview, len(content), confidence_score))
            conn.commit()
        
        # Store in ChromaDB if available
        if CHROMADB_AVAILABLE and hasattr(self, 'documents_collection') and self.documents_collection:
            try:
                self.documents_collection.add(
                    documents=[content],
                    metadatas=[{
                        "source_file": source_file,
                        "document_type": document_type,
                        "timestamp": timestamp or "",
                        "sender": sender or "",
                        "confidence_score": confidence_score
                    }],
                    ids=[doc_id]
                )
            except Exception as e:
                self.logger.warning(f"ChromaDB storage failed: {e}")
        
        self.logger.info(f"Stored document {doc_id} from {source_file}")
        return doc_id
    
    def store_analysis_result(self, document_id: str, analysis_type: str, 
                            prompt_used: str, ai_response: str, 
                            confidence_score: float, relevance_score: float,
                            legal_facts: Optional[List[str]] = None, legal_laws: Optional[List[str]] = None,
                            firecrawl_triggered: bool = False, firecrawl_results: Optional[str] = None,
                            processing_time: float = 0.0) -> str:
        """Store analysis results"""
        
        analysis_id = str(uuid.uuid4())
        
        # Store in SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_results 
                (id, document_id, analysis_type, prompt_used, ai_response, 
                 confidence_score, relevance_score, legal_facts, legal_laws,
                 firecrawl_triggered, firecrawl_results, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (analysis_id, document_id, analysis_type, prompt_used, ai_response,
                  confidence_score, relevance_score, 
                  json.dumps(legal_facts) if legal_facts else None,
                  json.dumps(legal_laws) if legal_laws else None,
                  firecrawl_triggered, firecrawl_results, processing_time))
            conn.commit()
        
        # Store in ChromaDB for semantic search if available
        if CHROMADB_AVAILABLE and hasattr(self, 'analysis_collection') and self.analysis_collection:
            try:
                self.analysis_collection.add(
                    documents=[ai_response],
                    metadatas=[{
                        "document_id": document_id,
                        "analysis_type": analysis_type,
                        "confidence_score": confidence_score,
                        "relevance_score": relevance_score,
                        "firecrawl_triggered": firecrawl_triggered
                    }],
                    ids=[analysis_id]
                )
            except Exception as e:
                self.logger.warning(f"ChromaDB analysis storage failed: {e}")
        
        return analysis_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        return None

    def get_documents_for_analysis(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents that need analysis for Phase 2"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get documents that haven't been analyzed yet
            query = """
                SELECT d.* FROM documents d 
                LEFT JOIN analysis_results ar ON d.id = ar.document_id 
                WHERE ar.document_id IS NULL OR d.processing_status = 'pending'
                ORDER BY d.created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if rows:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        return []
    
    def save_ui_state(self, prompt: Optional[str] = None, output: Optional[str] = None, 
                     settings: Optional[Dict] = None, session_history: Optional[List] = None):
        """Save UI state for persistence"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current state
            cursor.execute("SELECT * FROM ui_state WHERE id = 'current'")
            current = cursor.fetchone()
            
            if current:
                # Update existing
                cursor.execute("""
                    UPDATE ui_state SET 
                    last_prompt = COALESCE(?, last_prompt),
                    last_output = COALESCE(?, last_output),
                    current_settings = COALESCE(?, current_settings),
                    session_history = COALESCE(?, session_history),
                    updated_at = CURRENT_TIMESTAMP
                    WHERE id = 'current'
                """, (prompt, output, 
                      json.dumps(settings) if settings else None,
                      json.dumps(session_history) if session_history else None))
            else:
                # Create new
                cursor.execute("""
                    INSERT INTO ui_state (id, last_prompt, last_output, current_settings, session_history)
                    VALUES ('current', ?, ?, ?, ?)
                """, (prompt, output,
                      json.dumps(settings) if settings else None,
                      json.dumps(session_history) if session_history else None))
            
            conn.commit()
    
    def load_ui_state(self) -> Dict[str, Any]:
        """Load UI state"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ui_state WHERE id = 'current'")
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                state = dict(zip(columns, row))
                
                # Parse JSON fields
                if state.get('current_settings'):
                    state['current_settings'] = json.loads(state['current_settings'])
                if state.get('session_history'):
                    state['session_history'] = json.loads(state['session_history'])
                
                return state
            
        return {
            'last_prompt': '',
            'last_output': '',
            'current_settings': {},
            'session_history': []
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics for the UI"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Document stats
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'completed'")
            processed_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            total_analyses = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_results WHERE firecrawl_triggered = TRUE")
            firecrawl_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(confidence_score) FROM analysis_results WHERE confidence_score > 0")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            return {
                'total_documents': total_docs,
                'processed_documents': processed_docs,
                'total_analyses': total_analyses,
                'firecrawl_searches': firecrawl_count,
                'average_confidence': round(avg_confidence, 2),
                'processing_percentage': round((processed_docs / total_docs * 100) if total_docs > 0 else 0, 1)
            }

    def get_document_stats(self):
        """Get basic document statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count documents by type
                cursor.execute("SELECT document_type, COUNT(*) FROM documents GROUP BY document_type")
                doc_types = cursor.fetchall()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM documents")
                total_docs = cursor.fetchone()[0]
                
                # Character count stats
                cursor.execute("SELECT SUM(character_count), AVG(character_count) FROM documents")
                char_stats = cursor.fetchone()
                
                # Analysis status
                cursor.execute("SELECT processing_status, COUNT(*) FROM documents GROUP BY processing_status")
                status_counts = cursor.fetchall()
                
                return {
                    "total_documents": total_docs,
                    "document_types": dict(doc_types),
                    "total_characters": char_stats[0] or 0,
                    "avg_characters": round(char_stats[1] or 0, 2),
                    "processing_status": dict(status_counts)
                }
        except Exception as e:
            self.logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}

    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search across documents"""
        if not CHROMADB_AVAILABLE or not hasattr(self, 'documents_collection') or not self.documents_collection:
            return []
            
        try:
            results = self.documents_collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                doc = self.get_document(doc_id)
                if doc:
                    # Add similarity score if distances are available
                    try:
                        distances = results.get('distances')
                        if distances and isinstance(distances, list) and len(distances) > 0 and len(distances[0]) > i:
                            doc['similarity_score'] = 1 - distances[0][i]  # Convert distance to similarity
                        else:
                            doc['similarity_score'] = 0.5  # Default similarity
                    except (KeyError, IndexError, TypeError):
                        doc['similarity_score'] = 0.5  # Default similarity on any error
                    documents.append(doc)
            
            return documents
        except Exception as e:
            self.logger.warning(f"Document search failed: {e}")
            return []
    
    def search_analysis(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search across analysis results"""
        if not CHROMADB_AVAILABLE or not hasattr(self, 'analysis_collection') or not self.analysis_collection:
            return []
            
        try:
            results = self.analysis_collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            analyses = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for i, analysis_id in enumerate(results['ids'][0]):
                    cursor.execute("SELECT * FROM analysis_results WHERE id = ?", (analysis_id,))
                    row = cursor.fetchone()
                    if row:
                        columns = [desc[0] for desc in cursor.description]
                        analysis = dict(zip(columns, row))
                        # Add similarity score if distances are available
                        try:
                            distances = results.get('distances')
                            if distances and isinstance(distances, list) and len(distances) > 0 and len(distances[0]) > i:
                                analysis['similarity_score'] = 1 - distances[0][i]
                            else:
                                analysis['similarity_score'] = 0.5  # Default similarity
                        except (KeyError, IndexError, TypeError):
                            analysis['similarity_score'] = 0.5  # Default similarity on any error
                        analyses.append(analysis)
            
            return analyses
        except Exception as e:
            self.logger.warning(f"Analysis search failed: {e}")
            return []

    def store_document_with_analysis(self, source_file: str, document_type: str, content: str, 
                                   timestamp: Optional[str] = None, sender: Optional[str] = None, 
                                   analysis_engine=None, baseline_content: Optional[str] = None) -> str:
        """Store a document and automatically perform AI analysis"""
        import time
        start_time = time.time()
        doc_id = self.store_document(source_file, document_type, content, timestamp, sender)
        if analysis_engine and len(content.strip()) > 50:
            try:
                self.logger.info(f"🧠 Starting AI analysis for document {doc_id}")
                analysis_result = analysis_engine.analyze_document(content, "comprehensive", baseline_content)
                analysis_id = self.store_analysis_result(
                    document_id=doc_id,
                    analysis_type=analysis_result.analysis_type,
                    prompt_used=f"Comprehensive legal analysis - {analysis_result.model_used}",
                    ai_response=analysis_result.summary,
                    confidence_score=analysis_result.confidence_score,
                    relevance_score=analysis_result.relevance_score,
                    legal_facts=analysis_result.key_facts,
                    legal_laws=analysis_result.legal_issues,
                    firecrawl_triggered=analysis_result.needs_legal_research,
                    firecrawl_results=json.dumps({
                        "research_topics": analysis_result.research_topics,
                        "legal_citations_needed": analysis_result.legal_citations_needed,
                        "jurisdiction": analysis_result.jurisdiction,
                        "evidence_value": analysis_result.evidence_value,
                        "mentions_property": analysis_result.mentions_property,
                        "mentions_inheritance": analysis_result.mentions_inheritance,
                        "mentions_wills": analysis_result.mentions_wills,
                        "mentions_tax": analysis_result.mentions_tax,
                        "mentions_domicile": analysis_result.mentions_domicile,
                        "timeline_relevance": analysis_result.timeline_relevance,
                        "chronological_context": analysis_result.chronological_context,
                        "contradictions": analysis_result.contradictions
                    }) if analysis_result.needs_legal_research else None,
                    processing_time=analysis_result.processing_time
                )
                self._update_document_confidence(doc_id, analysis_result.confidence_score)
                processing_time = time.time() - start_time
                self.logger.info(f"✅ Document and analysis stored: {doc_id} (total: {processing_time:.2f}s)")
            except Exception as e:
                self.logger.error(f"❌ Analysis failed for document {doc_id}: {e}")
        return doc_id

    def analyze_documents_phase2(self, analysis_engine, firecrawl_client=None, 
                               baseline_path: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Phase 2: Individual AI Analysis of all documents
        Analyzes each document with Gemini AI using baseline context and triggers Firecrawl when needed
        """
        import time
        start_time = time.time()
        baseline_content = None
        if baseline_path:
            baseline_content = self.load_baseline_document(baseline_path)
            if baseline_content:
                self.logger.info(f"✅ Loaded baseline document: {len(baseline_content)} characters")
            else:
                self.logger.warning("⚠️ Could not load baseline document")
        documents = self.get_documents_for_analysis(limit)
        if not documents:
            self.logger.info("📄 No documents need analysis")
            return {
                'total_documents': 0,
                'analyzed_documents': 0,
                'failed_documents': 0,
                'firecrawl_searches': 0,
                'processing_time': 0.0
            }
        self.logger.info(f"🚀 Starting Phase 2 AI Analysis: {len(documents)} documents")
        results = {
            'total_documents': len(documents),
            'analyzed_documents': 0,
            'failed_documents': 0,
            'firecrawl_searches': 0,
            'processing_time': 0.0,
            'analysis_details': []
        }
        for i, doc in enumerate(documents):
            doc_id = doc['id']
            source_file = doc['source_file']
            content = doc.get('content', '')
            self.logger.info(f"📊 Analyzing {i+1}/{len(documents)}: {source_file}")
            if not content or len(content.strip()) < 20:
                self.logger.warning(f"⚠️ Skipping document {doc_id} - insufficient content")
                results['failed_documents'] += 1
                continue
            try:
                doc_start_time = time.time()
                analysis_result = analysis_engine.analyze_document(
                    content, 
                    "comprehensive", 
                    baseline_content
                )
                analysis_id = self.store_analysis_result(
                    document_id=doc_id,
                    analysis_type=analysis_result.analysis_type,
                    prompt_used=f"Phase 2 Comprehensive Analysis with baseline context",
                    ai_response=analysis_result.summary,
                    confidence_score=analysis_result.confidence_score,
                    relevance_score=analysis_result.relevance_score,
                    legal_facts=analysis_result.key_facts,
                    legal_laws=analysis_result.legal_issues,
                    firecrawl_triggered=analysis_result.needs_legal_research,
                    firecrawl_results=json.dumps({
                        "research_topics": analysis_result.research_topics,
                        "legal_citations_needed": analysis_result.legal_citations_needed,
                        "jurisdiction": analysis_result.jurisdiction,
                        "evidence_value": analysis_result.evidence_value,
                        "mentions_property": analysis_result.mentions_property,
                        "mentions_inheritance": analysis_result.mentions_inheritance,
                        "mentions_wills": analysis_result.mentions_wills,
                        "mentions_tax": analysis_result.mentions_tax,
                        "mentions_domicile": analysis_result.mentions_domicile,
                        "timeline_relevance": analysis_result.timeline_relevance,
                        "chronological_context": analysis_result.chronological_context,
                        "contradictions": analysis_result.contradictions
                    }),
                    processing_time=analysis_result.processing_time
                )
                if analysis_result.needs_legal_research and firecrawl_client and analysis_result.research_topics:
                    self.logger.info(f"🌐 Triggering Firecrawl research for {len(analysis_result.research_topics)} topics")
                    for topic in analysis_result.research_topics[:3]:
                        try:
                            firecrawl_result = firecrawl_client.search_legal_resources(topic)
                            if firecrawl_result:
                                results['firecrawl_searches'] += 1
                                self.logger.info(f"✅ Firecrawl search completed for: {topic}")
                        except Exception as fc_error:
                            self.logger.error(f"❌ Firecrawl search failed for {topic}: {fc_error}")
                self._update_document_analysis_complete(doc_id, analysis_result.confidence_score)
                doc_time = time.time() - doc_start_time
                results['analyzed_documents'] += 1
                results['analysis_details'].append({
                    'document_id': doc_id,
                    'source_file': source_file,
                    'confidence_score': analysis_result.confidence_score,
                    'relevance_score': analysis_result.relevance_score,
                    'evidence_value': analysis_result.evidence_value,
                    'needs_research': analysis_result.needs_legal_research,
                    'processing_time': doc_time
                })
                self.logger.info(f"✅ Analysis complete: {source_file} (confidence: {analysis_result.confidence_score:.2f}, {doc_time:.2f}s)")
            except Exception as e:
                self.logger.error(f"❌ Analysis failed for {source_file}: {e}")
                results['failed_documents'] += 1
                self._update_document_analysis_failed(doc_id, str(e))
        results['processing_time'] = time.time() - start_time
        self.logger.info(f"🎉 Phase 2 Analysis Complete: {results['analyzed_documents']}/{results['total_documents']} documents analyzed in {results['processing_time']:.2f}s")
        return results

    def run_deep_research_phase3(self, analysis_engine, baseline_path: Optional[str] = None, session_name: str = "Deep Research Session") -> Dict[str, Any]:
        """
        Phase 3: Deep Research & Cross-Referencing
        Aggregates all AI insights, finds cross-document patterns, contradictions, and timeline events.
        Stores results in the deep_research table.
        """
        import time
        start_time = time.time()
        # Load all analysis results
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_results WHERE analysis_type != 'failed'")
            analyses = [dict(zip([desc[0] for desc in cursor.description], row)) for row in cursor.fetchall()]
        if not analyses:
            self.logger.warning("No analysis results found for deep research phase.")
            return {"status": "no_data"}
        # Aggregate all snippets and findings
        all_snippets = [a['ai_response'] for a in analyses if a.get('ai_response')]
        all_facts = []
        all_issues = []
        for a in analyses:
            try:
                facts = json.loads(a.get('legal_facts', '[]'))
                issues = json.loads(a.get('legal_laws', '[]'))
                all_facts.extend(facts)
                all_issues.extend(issues)
            except Exception:
                continue
        # Compose a research query for Gemini
        research_query = (
            "Summarize the key legal facts, contradictions, and timeline events across all analyzed documents. "
            "Highlight any cross-document patterns, inconsistencies, or research needs. "
            "Provide a structured JSON with: summary, contradictions, timeline_events, research_gaps."
        )
        context = "\n".join(all_snippets[:30])  # Limit context size
        # Use baseline doc if available
        baseline_content = self.load_baseline_document(baseline_path) if baseline_path else None
        prompt = f"{research_query}\n\nContext:\n{context}\n\nBaseline Reference:\n{baseline_content or ''}"
        import asyncio
        try:
            ai_result = analysis_engine.analyze_document(prompt, analysis_type="deep_research", baseline_content=baseline_content)
            # Store in deep_research table
            session_id = str(uuid.uuid4())
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO research_sessions (id, session_name, prompt_template, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (session_id, session_name, research_query, 'completed'))
                cursor.execute("""
                    INSERT INTO deep_research (id, session_id, research_query, related_documents, findings, contradictions, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    session_id,
                    research_query,
                    json.dumps([a['document_id'] for a in analyses]),
                    ai_result.summary,
                    json.dumps(ai_result.contradictions),
                    ai_result.confidence_score
                ))
                conn.commit()
            return {"status": "success", "summary": ai_result.summary, "contradictions": ai_result.contradictions}
        except Exception as e:
            self.logger.error(f"Deep research phase failed: {e}")
            return {"status": "error", "error": str(e)}

    def build_timeline_phase4(self, analysis_engine, baseline_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 4: Timeline & Fact Synthesis
        Builds a chronological timeline from all document analyses and extracts key facts, events, and actors.
        """
        import time
        start_time = time.time()
        
        # Get all analysis results with timeline relevance
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ar.*, d.source_file, d.timestamp, d.sender, d.document_type 
                FROM analysis_results ar 
                JOIN documents d ON ar.document_id = d.id 
                WHERE ar.analysis_type != 'failed'
                ORDER BY d.timestamp, d.created_at
            """)
            analyses = [dict(zip([desc[0] for desc in cursor.description], row)) for row in cursor.fetchall()]
        
        if not analyses:
            self.logger.warning("No analysis results found for timeline phase.")
            return {"status": "no_data"}
        
        # Extract timeline events and chronological context
        timeline_events = []
        key_actors = set()
        legal_deadlines = []
        
        for analysis in analyses:
            try:
                firecrawl_data = json.loads(analysis.get('firecrawl_results', '{}'))
                if firecrawl_data.get('timeline_relevance'):
                    event = {
                        'document_id': analysis['document_id'],
                        'source_file': analysis['source_file'],
                        'timestamp': analysis['timestamp'],
                        'sender': analysis['sender'],
                        'chronological_context': firecrawl_data.get('chronological_context', ''),
                        'confidence_score': analysis['confidence_score'],
                        'summary': analysis['ai_response'][:200]
                    }
                    timeline_events.append(event)
                
                # Extract actors from sender and legal facts
                if analysis['sender']:
                    key_actors.add(analysis['sender'])
                
                legal_facts = json.loads(analysis.get('legal_facts', '[]'))
                for fact in legal_facts:
                    # Simple extraction of names (could be enhanced)
                    import re
                    names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', fact)
                    key_actors.update(names)
                    
                    # Look for dates and deadlines
                    if any(word in fact.lower() for word in ['deadline', 'due', 'expire', 'must be']):
                        legal_deadlines.append({
                            'deadline_text': fact,
                            'document_id': analysis['document_id'],
                            'confidence': analysis['confidence_score']
                        })
                        
            except Exception as e:
                self.logger.warning(f"Error processing analysis {analysis['id']}: {e}")
                continue
        
        # Use AI to synthesize timeline and identify gaps
        timeline_prompt = f"""
        Analyze this legal case timeline and provide a structured JSON response:
        
        Timeline Events: {json.dumps(timeline_events, indent=2)}
        Key Actors: {list(key_actors)}
        Legal Deadlines: {json.dumps(legal_deadlines, indent=2)}
        
        Provide JSON with:
        - chronological_summary: Brief timeline overview
        - critical_dates: Key dates and their significance
        - timeline_gaps: Missing periods or information
        - actor_relationships: Relationships between key people
        - deadline_analysis: Assessment of legal deadlines
        - inconsistencies: Timeline contradictions
        """
        
        baseline_content = self.load_baseline_document(baseline_path) if baseline_path else None
        
        try:
            timeline_result = analysis_engine.analyze_document(timeline_prompt, "timeline_synthesis", baseline_content)
            
            # Store timeline results
            session_id = str(uuid.uuid4())
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO research_sessions (id, session_name, prompt_template, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (session_id, "Timeline & Fact Synthesis", timeline_prompt[:500], 'completed'))
                
                cursor.execute("""
                    INSERT INTO deep_research (id, session_id, research_query, related_documents, findings, contradictions, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    session_id,
                    "Timeline and fact synthesis",
                    json.dumps([a['document_id'] for a in analyses]),
                    timeline_result.summary,
                    json.dumps(timeline_result.contradictions),
                    timeline_result.confidence_score
                ))
                conn.commit()
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Timeline synthesis completed in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "timeline_events": len(timeline_events),
                "key_actors": len(key_actors),
                "legal_deadlines": len(legal_deadlines),
                "synthesis": timeline_result.summary,
                "contradictions": timeline_result.contradictions,
                "confidence_score": timeline_result.confidence_score,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Timeline synthesis failed: {e}")
            return {"status": "error", "error": str(e)}

    def generate_comprehensive_report_phase5(self, analysis_engine, baseline_path: Optional[str] = None, 
                                           report_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Phase 5: Comprehensive Report Generation
        Generates a final legal report using all previous phases' analyses.
        """
        import time
        start_time = time.time()
        
        # Gather all data from previous phases
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Document statistics
            cursor.execute("SELECT COUNT(*) as total, AVG(confidence_score) as avg_confidence FROM documents")
            doc_stats = cursor.fetchone()
            
            # Analysis results
            cursor.execute("SELECT COUNT(*) as total, AVG(confidence_score) as avg_confidence FROM analysis_results WHERE analysis_type != 'failed'")
            analysis_stats = cursor.fetchone()
            
            # Deep research results
            cursor.execute("SELECT * FROM deep_research ORDER BY created_at DESC LIMIT 10")
            deep_research = [dict(zip([desc[0] for desc in cursor.description], row)) for row in cursor.fetchall()]
            
            # High confidence findings
            cursor.execute("""
                SELECT ar.ai_response, ar.confidence_score, ar.relevance_score, d.source_file
                FROM analysis_results ar 
                JOIN documents d ON ar.document_id = d.id 
                WHERE ar.confidence_score > 0.7 AND ar.analysis_type != 'failed'
                ORDER BY ar.confidence_score DESC LIMIT 20
            """)
            high_confidence_findings = [dict(zip([desc[0] for desc in cursor.description], row)) for row in cursor.fetchall()]
            
            # Research needs
            cursor.execute("""
                SELECT ar.firecrawl_results, d.source_file 
                FROM analysis_results ar 
                JOIN documents d ON ar.document_id = d.id 
                WHERE ar.firecrawl_triggered = TRUE
            """)
            research_needs_raw = cursor.fetchall()
        
        # Process research needs
        research_topics = set()
        for row in research_needs_raw:
            try:
                firecrawl_data = json.loads(row[0] or '{}')
                topics = firecrawl_data.get('research_topics', [])
                research_topics.update(topics)
            except:
                continue
        
        # Compile report data
        report_data = {
            "case_overview": {
                "total_documents": doc_stats[0] if doc_stats else 0,
                "average_document_confidence": round(doc_stats[1] or 0, 2),
                "total_analyses": analysis_stats[0] if analysis_stats else 0,
                "average_analysis_confidence": round(analysis_stats[1] or 0, 2),
                "deep_research_sessions": len(deep_research)
            },
            "high_confidence_findings": [
                {
                    "summary": finding["ai_response"][:300],
                    "confidence": finding["confidence_score"],
                    "relevance": finding["relevance_score"],
                    "source": finding["source_file"]
                }
                for finding in high_confidence_findings
            ],
            "research_recommendations": list(research_topics),
            "deep_insights": [
                {
                    "findings": research["findings"],
                    "contradictions": json.loads(research.get("contradictions", "[]")),
                    "confidence": research["confidence_score"]
                }
                for research in deep_research
            ]
        }
        
        # Generate AI report
        report_prompt = f"""
        Generate a comprehensive legal case report based on the following analysis data:
        
        {json.dumps(report_data, indent=2)}
        
        Structure the report as JSON with:
        - executive_summary: Key findings and recommendations
        - case_strength_assessment: Overall case strength (1-10) with reasoning
        - critical_evidence: Most important evidence pieces
        - legal_vulnerabilities: Potential weaknesses or risks
        - timeline_assessment: Key timeline events and gaps
        - recommended_actions: Next steps and legal strategy
        - research_priorities: Most important research needs
        - confidence_analysis: Assessment of evidence reliability
        
        Focus on inheritance/probate law issues, property disputes, and jurisdictional concerns.
        """
        
        baseline_content = self.load_baseline_document(baseline_path) if baseline_path else None
        
        try:
            report_result = analysis_engine.analyze_document(report_prompt, "comprehensive_report", baseline_content)
            
            # Store report
            session_id = str(uuid.uuid4())
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO research_sessions (id, session_name, prompt_template, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (session_id, f"Comprehensive Report - {report_type}", report_prompt[:500], 'completed'))
                
                cursor.execute("""
                    INSERT INTO deep_research (id, session_id, research_query, related_documents, findings, contradictions, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    session_id,
                    f"Comprehensive case report - {report_type}",
                    json.dumps([f["source"] for f in high_confidence_findings]),
                    report_result.summary,
                    json.dumps(report_result.contradictions),
                    report_result.confidence_score
                ))
                conn.commit()
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Comprehensive report generated in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "report": report_result.summary,
                "confidence_score": report_result.confidence_score,
                "case_statistics": report_data["case_overview"],
                "research_recommendations": list(research_topics),
                "processing_time": processing_time,
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_case_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the entire case analysis status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Phase completion status
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'completed'")
            phase2_completed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM research_sessions WHERE session_name LIKE '%Deep Research%'")
            phase3_completed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM research_sessions WHERE session_name LIKE '%Timeline%'")
            phase4_completed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM research_sessions WHERE session_name LIKE '%Comprehensive Report%'")
            phase5_completed = cursor.fetchone()[0]
            
            # Latest insights
            cursor.execute("""
                SELECT ar.ai_response, ar.confidence_score, d.source_file 
                FROM analysis_results ar 
                JOIN documents d ON ar.document_id = d.id 
                WHERE ar.confidence_score > 0.8 
                ORDER BY ar.created_at DESC LIMIT 5
            """)
            top_insights = [dict(zip([desc[0] for desc in cursor.description], row)) for row in cursor.fetchall()]
            
            return {
                "phase_status": {
                    "phase1_ingestion": f"{total_docs} documents ingested",
                    "phase2_analysis": f"{phase2_completed}/{total_docs} documents analyzed",
                    "phase3_deep_research": f"{phase3_completed} sessions completed",
                    "phase4_timeline": f"{phase4_completed} timeline syntheses",
                    "phase5_reports": f"{phase5_completed} comprehensive reports"
                },
                "top_insights": [
                    {
                        "summary": insight['ai_response'][:200] + "..." if len(insight['ai_response']) > 200 else insight['ai_response'],
                        "confidence": insight['confidence_score'],
                        "source": insight['source_file']
                    } for insight in top_insights
                ],
                "recommendations": self._get_next_phase_recommendation(
                    total_docs, phase2_completed, phase3_completed, phase4_completed, phase5_completed
                )
            }

    def _get_next_phase_recommendation(self, total_docs: int, phase2: int, phase3: int, phase4: int, phase5: int) -> str:
        """Determine what phase should be executed next"""
        if phase2 < total_docs:
            return f"Phase 2: Continue AI analysis ({phase2}/{total_docs} completed)"
        elif phase3 == 0:
            return "Phase 3: Start deep research & cross-referencing"
        elif phase4 == 0:
            return "Phase 4: Begin timeline & fact synthesis"
        elif phase5 == 0:
            return "Phase 5: Generate comprehensive report"
        else:
            return "Analysis complete: All phases finished"

    def load_baseline_document(self, baseline_path: str) -> Optional[str]:
        """Load baseline document content from file path"""
        try:
            if baseline_path and Path(baseline_path).exists():
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                self.logger.warning(f"Baseline document not found: {baseline_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading baseline document: {e}")
            return None

    def _update_document_confidence(self, doc_id: str, confidence_score: float):
        """Update document confidence score"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE documents SET confidence_score = ? WHERE id = ?",
                (confidence_score, doc_id)
            )

    def _update_document_analysis_complete(self, doc_id: str, confidence_score: float):
        """Mark document analysis as complete"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE documents SET processing_status = 'completed', confidence_score = ? WHERE id = ?",
                (confidence_score, doc_id)
            )

    def _update_document_analysis_failed(self, doc_id: str, error_msg: str):
        """Mark document analysis as failed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE documents SET processing_status = 'failed' WHERE id = ?",
                (doc_id,)
            )

    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analyses for UI display"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        a.id,
                        a.document_id,
                        d.source_file as document_name,
                        COALESCE(d.document_type, 'Unknown') as document_type,
                        a.confidence_score,
                        a.relevance_score,
                        a.legal_facts,
                        a.processing_time,
                        a.created_at,
                        a.analysis_type
                    FROM analysis_results a
                    JOIN documents d ON a.document_id = d.id
                    ORDER BY a.created_at DESC
                    LIMIT ?
                """
                
                cursor.execute(query, (limit,))
                results = cursor.fetchall()
                
                analyses = []
                for row in results:
                    analyses.append({
                        'id': row[0],
                        'document_id': row[1],
                        'document_name': row[2],
                        'document_type': row[3],
                        'confidence_score': row[4],
                        'relevance_score': row[5],
                        'legal_facts': row[6],
                        'processing_time': row[7],
                        'created_at': row[8],
                        'analysis_type': row[9]
                    })
                
                return analyses
                
        except Exception as e:
            print(f"Error getting recent analyses: {e}")
            return []
