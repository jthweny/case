import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional
import json
import time
import os
import sys
import logging
import traceback
import sqlite3
from pathlib import Path
import hashlib
from typing import List, Dict, Any

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/streamlit_debug.log')
    ]
)

# Create logger
logger = logging.getLogger("StreamlitUI")

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
try:
    from src.hybrid_database import HybridDatabaseManager
    from src.document_parser import DocumentParser
    from src.parallel_gemini_api import ParallelGeminiAPIManager
    from src.firecrawl_client import FirecrawlClient
    from enhanced_phase2_interface import EnhancedPhase2Interface
    logger.info("✅ All modules imported successfully")
    # Note: legal_researcher has compatibility issues, importing individually as needed
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    st.error(f"Import error: {e}. Please ensure all dependencies are installed.")
    st.stop()

class LegalCaseAnalysisUI:
    """
    Comprehensive Streamlit UI for Legal Case Analysis System
    """
    
    def __init__(self):
        self.db = None
        self.api_manager = None
        self.doc_parser = None
        self.researcher = None
        self.processor = None
        self._init_session_state()
        self._init_components()
        
        # Initialize performance optimization
        self._init_performance_cache()
    
    def _init_session_state(self):
        """Initialize Streamlit session state with Gemini 2.5 Pro optimizations"""
        # UI state persistence
        if 'ui_initialized' not in st.session_state:
            st.session_state.ui_initialized = True
            st.session_state.last_prompt = ""
            st.session_state.last_output = ""
            st.session_state.session_history = []
            st.session_state.current_settings = {
                'max_tokens': 4096,  # Increased for Gemini 2.5 Pro
                'temperature': 0.1,  # Lower for legal analysis consistency
                'prefer_preview_keys': True,
                'enable_firecrawl': True,
                'case_docs_path': '/home/joshuathweny/Desktop/case docs ONLY',
                'baseline_pdf_path': '/home/joshuathweny/Desktop/case docs ONLY/my perspective/REFINED_COMPREHENSIVE_LEGAL_ANALYSIS (1).pdf'
            }
        
        # Gemini 2.5 Pro session state management
        if 'gemini_25_client' not in st.session_state:
            st.session_state.gemini_25_client = None
            st.session_state.api_connectivity_status = None
            st.session_state.last_connectivity_check = None
            st.session_state.current_model = None
        
        # Processing state
        if 'processing_active' not in st.session_state:
            st.session_state.processing_active = False
            st.session_state.current_phase = 'idle'  # idle, ingestion, analysis, deep_research
            st.session_state.processed_count = 0
            st.session_state.total_count = 0
            st.session_state.current_document = None
    
    def _init_components(self):
        """Initialize backend components"""
        try:
            logger.info("🔧 Initializing backend components...")
            
            logger.info("📊 Initializing database manager...")
            self.db = HybridDatabaseManager()
            logger.info("✅ Database manager initialized")
            
            logger.info("🔑 Initializing parallel Gemini API manager...")
            
            # Initialize API manager and store in session state for consistency
            if 'api_manager' not in st.session_state or st.session_state.api_manager is None:
                self.api_manager = ParallelGeminiAPIManager()
                st.session_state.api_manager = self.api_manager
                st.session_state.gemini_25_client = self.api_manager
            else:
                self.api_manager = st.session_state.api_manager
            
            logger.info("✅ Parallel Gemini API manager initialized")
            
            logger.info("📄 Initializing document parser...")
            self.doc_parser = DocumentParser()
            logger.info("✅ Document parser initialized")
            
            logger.info("🌐 Initializing Firecrawl client...")
            self.firecrawl_client = FirecrawlClient()
            logger.info("✅ Firecrawl client initialized")
            
            logger.info("🧠 Initializing enhanced legal analysis engine...")
            sys.path.append('.')
            from enhanced_legal_analyzer import EnhancedLegalAnalyzer
            self.analysis_engine = EnhancedLegalAnalyzer()
            logger.info("✅ Enhanced legal analysis engine initialized")
            
            logger.info("🎯 Initializing enhanced Phase 2 interface...")
            self.enhanced_phase2 = EnhancedPhase2Interface()
            logger.info("✅ Enhanced Phase 2 interface initialized")
            
            # Load baseline document for context
            self._load_baseline_document()
            
            # Load UI state from database
            logger.info("💾 Loading UI state from database...")
            saved_state = self.db.load_ui_state()
            if saved_state:
                st.session_state.last_prompt = saved_state.get('last_prompt', '')
                st.session_state.last_output = saved_state.get('last_output', '')
                if saved_state.get('current_settings'):
                    st.session_state.current_settings.update(saved_state['current_settings'])
                if saved_state.get('session_history'):
                    st.session_state.session_history = saved_state['session_history']
                logger.info("✅ UI state loaded successfully")
            else:
                logger.info("ℹ️  No previous UI state found")
            
            logger.info("🎉 All components initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize components: {e}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            st.error("Check the terminal/logs for detailed error information.")
            st.stop()
    
    def _load_baseline_document(self):
        """Load the baseline refined document for analysis context"""
        try:
            baseline_path = st.session_state.current_settings.get('baseline_pdf_path', '')
            if baseline_path and Path(baseline_path).exists():
                logger.info(f"📖 Loading baseline document from: {baseline_path}")
                
                # Parse the baseline document
                baseline_docs = self.doc_parser.get_all_documents(str(Path(baseline_path).parent))
                baseline_content = ""
                
                for doc in baseline_docs:
                    if doc['source_file'] == baseline_path:
                        baseline_content = doc['content']
                        break
                
                if baseline_content:
                    # Store baseline content for analysis context
                    st.session_state.baseline_content = baseline_content[:8000]  # Limit size
                    logger.info(f"✅ Baseline document loaded: {len(baseline_content)} characters")
                else:
                    logger.warning("⚠️ Baseline document found but no content extracted")
                    st.session_state.baseline_content = ""
            else:
                logger.warning(f"⚠️ Baseline document not found at: {baseline_path}")
                st.session_state.baseline_content = ""
                
        except Exception as e:
            logger.error(f"❌ Failed to load baseline document: {e}")
            st.session_state.baseline_content = ""
    
    def _init_performance_cache(self):
        """Initialize performance optimization and caching"""
        self.cache_file = "data/streamlit_cache.json"
        self.processed_docs = set()
        self.processing_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "processing_time": 0
        }
        self._load_performance_cache()
    
    def _load_performance_cache(self):
        """Load performance cache for development efficiency"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.processed_docs = set(cache_data.get("processed_docs", []))
                    self.processing_stats = cache_data.get("processing_stats", self.processing_stats)
                    logger.info(f"📋 Loaded cache: {len(self.processed_docs)} processed documents")
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
    
    def _save_performance_cache(self):
        """Save performance cache for next session"""
        try:
            Path(self.cache_file).parent.mkdir(exist_ok=True)
            cache_data = {
                "processed_docs": list(self.processed_docs),
                "processing_stats": self.processing_stats,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def _get_document_key(self, doc: Dict[str, Any]) -> str:
        """Generate unique key for document caching"""
        content = str(doc.get('content', ''))
        source = doc.get('source_file', 'unknown')
        return f"{source}_{len(content)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
    
    def _filter_cached_documents(self, documents: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """Filter out already processed documents"""
        new_docs = []
        cached_count = 0
        
        for doc in documents:
            doc_key = self._get_document_key(doc)
            if doc_key not in self.processed_docs:
                new_docs.append(doc)
            else:
                cached_count += 1
        
        return new_docs, cached_count
    
    def _mark_document_processed(self, doc: Dict[str, Any]):
        """Mark document as processed in cache"""
        doc_key = self._get_document_key(doc)
        self.processed_docs.add(doc_key)
    
    def _explain_document_count(self, total_count: int) -> str:
        """Explain why document count might be high"""
        if total_count > 1000:
            return f"""
            ### 📊 Document Count Explanation (Total: {total_count:,})
            
            **✅ This high count is normal and beneficial:**
            
            - **Timeline files**: Parsed into individual chronological entries (~2,000-3,000 entries)
            - **MBOX email files**: Split into individual messages (~10-50 per file)
            - **Each entry**: Becomes a separate document for detailed AI analysis
            
            **🎯 Why this granular approach helps:**
            - Better chronological understanding of legal events
            - Individual analysis of each email/timeline entry
            - More precise legal insights per document segment
            - Enables comprehensive case timeline reconstruction
            
            **⚡ Performance**: Processing uses intelligent caching and batch operations.
            """
        return f"Processing {total_count:,} documents with optimization..."
    
    def save_ui_state(self):
        """Save current UI state to database"""
        try:
            self.db.save_ui_state(
                prompt=st.session_state.last_prompt,
                output=st.session_state.last_output,
                settings=st.session_state.current_settings,
                session_history=st.session_state.session_history
            )
        except Exception as e:
            st.warning(f"Failed to save UI state: {e}")
    
    def render_header(self):
        """Render the main header with real-time metrics"""
        st.set_page_config(
            page_title="Legal Case Analysis System",
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("⚖️ Legal Case Analysis System")
        st.markdown("---")
        
        # Create persistent placeholders for real-time updates
        if 'header_placeholders' not in st.session_state:
            st.session_state.header_placeholders = {}
        
        # Status bar
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'status_placeholder' not in st.session_state.header_placeholders:
                st.session_state.header_placeholders['status_placeholder'] = st.empty()
            self._update_status_metric()
        
        with col2:
            if 'documents_placeholder' not in st.session_state.header_placeholders:
                st.session_state.header_placeholders['documents_placeholder'] = st.empty()
            self._update_documents_metric()
        
        with col3:
            if 'analyses_placeholder' not in st.session_state.header_placeholders:
                st.session_state.header_placeholders['analyses_placeholder'] = st.empty()
            self._update_analyses_metric()
        
        with col4:
            if 'api_placeholder' not in st.session_state.header_placeholders:
                st.session_state.header_placeholders['api_placeholder'] = st.empty()
            self._update_api_metric()
    
    def _update_status_metric(self):
        """Update status metric in real-time"""
        with st.session_state.header_placeholders['status_placeholder']:
            if st.session_state.processing_active:
                st.metric("Status", "🟡 Processing", f"Phase: {st.session_state.current_phase}")
            else:
                st.metric("Status", "🟢 Ready", "Idle")
    
    def _update_documents_metric(self):
        """Update documents metric in real-time"""
        with st.session_state.header_placeholders['documents_placeholder']:
            stats = self.db.get_processing_stats()
            st.metric("Documents", stats['total_documents'], f"{stats['processing_percentage']}% processed")
    
    def _update_analyses_metric(self):
        """Update analyses metric in real-time"""
        with st.session_state.header_placeholders['analyses_placeholder']:
            stats = self.db.get_processing_stats()
            st.metric("Analyses", stats['total_analyses'], f"Avg confidence: {stats['average_confidence']}")
    
    def _update_api_metric(self):
        """Update API status metric in real-time"""
        with st.session_state.header_placeholders['api_placeholder']:
            api_status = self.api_manager.get_api_status()
            working_keys = api_status.get('working_keys', 0)
            total_keys = api_status.get('total_keys', 0)
            current_model = api_status.get('current_model', 'Unknown')
            st.metric("Gemini 2.5 Flash", f"{working_keys}/{total_keys} keys", f"Model: {current_model}")
    
    def refresh_header_metrics(self):
        """Refresh all header metrics in real-time"""
        if hasattr(st.session_state, 'header_placeholders'):
            self._update_status_metric()
            self._update_documents_metric()
            self._update_analyses_metric()
            self._update_api_metric()
    
    def render_sidebar(self):
        """Render the settings sidebar"""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Phase selection
            st.subheader("Processing Phase")
            phase = st.radio(
                "Select Phase:",
                ["Document Ingestion", "Individual Analysis", "Deep Research", "Report Generation"],
                help="Choose the current processing phase"
            )
            
            # API Settings
            st.subheader("AI Settings")
            st.session_state.current_settings['max_tokens'] = st.slider(
                "Max Output Tokens", 100, 8192, st.session_state.current_settings['max_tokens']
            )
            st.session_state.current_settings['temperature'] = st.slider(
                "Temperature", 0.0, 1.0, st.session_state.current_settings['temperature'], 0.1
            )
            st.session_state.current_settings['prefer_preview_keys'] = st.checkbox(
                "Prefer Preview Keys", st.session_state.current_settings['prefer_preview_keys']
            )
            
            # Research Settings
            st.subheader("Research Settings")
            st.session_state.current_settings['enable_firecrawl'] = st.checkbox(
                "Enable Firecrawl Research", st.session_state.current_settings['enable_firecrawl']
            )
            
            # File Paths
            st.subheader("File Paths")
            st.session_state.current_settings['case_docs_path'] = st.text_input(
                "Case Documents Directory",
                st.session_state.current_settings['case_docs_path']
            )
            st.session_state.current_settings['baseline_pdf_path'] = st.text_input(
                "Baseline PDF Path",
                st.session_state.current_settings['baseline_pdf_path']
            )
            
            # Save settings
            if st.button("💾 Save Settings"):
                self.save_ui_state()
                st.success("Settings saved!")
            
            # API Key Status
            st.subheader("API Key Status")
            if st.button("🔄 Refresh API Status"):
                self.api_manager.refresh_key_status()
                # Status will show updated info on next interaction
            
            api_status = self.api_manager.get_api_status()
            st.json(api_status)
    
    def render_api_key_configuration(self):
        """Render API key configuration interface"""
        st.subheader("🔑 API Key Configuration")
        
        # Check current API status
        try:
            api_status = self.api_manager.get_api_status()
            working_keys = api_status.get('working_keys', 0)
            total_keys = api_status.get('total_keys', 0)
            
            if working_keys > 0:
                st.success(f"✅ {working_keys}/{total_keys} Gemini 2.5 Pro API keys working")
                
                # Show provider status
                providers = api_status.get('providers', {})
                for provider, info in providers.items():
                    status_emoji = "✅" if info.get('status') == 'active' else "❌"
                    details = info.get('details', 'No details')
                    st.write(f"{status_emoji} **{provider}**: {info.get('status', 'unknown')} - {details}")
                
                # Show additional info
                st.info(f"🤖 **Current Model**: {api_status.get('current_model', 'Unknown')}")
                st.info(f"🔧 **SDK Version**: {api_status.get('sdk_version', 'Unknown')}")
                st.info(f"⏱️ **Timeout**: {api_status.get('timeout_ms', 0)/1000:.0f}s")
                
                return True
            else:
                st.error(f"❌ No working API keys found ({working_keys}/{total_keys})")
        
        except Exception as e:
            st.error(f"❌ API status check failed: {e}")
        
        # API Key Configuration Form
        st.markdown("---")
        st.markdown("### Configure API Keys")
        
        with st.form("api_key_form"):
            st.markdown("**Enter your API keys below:**")
            
            # Gemini API Key
            gemini_key = st.text_input(
                "🔮 Gemini API Key",
                type="password",
                help="Get your key from https://makersuite.google.com/app/apikey",
                placeholder="Enter your Gemini API key"
            )
            
            # OpenAI API Key
            openai_key = st.text_input(
                "🤖 OpenAI API Key", 
                type="password",
                help="Get your key from https://platform.openai.com/api-keys",
                placeholder="Enter your OpenAI API key"
            )
            
            # Submit button
            submitted = st.form_submit_button("💾 Save API Keys")
            
            if submitted:
                if gemini_key or openai_key:
                    # Save to environment (temporary for this session)
                    import os
                    if gemini_key:
                        os.environ['GEMINI_API_KEY'] = gemini_key
                        st.success("✅ Gemini API key saved for this session")
                    
                    if openai_key:
                        os.environ['OPENAI_API_KEY'] = openai_key
                        st.success("✅ OpenAI API key saved for this session")
                    
                    # Reinitialize API manager
                    try:
                        self.api_manager = SmartAPIManager()
                        st.success("🔄 API manager reinitialized")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to reinitialize API manager: {e}")
                else:
                    st.warning("⚠️ Please enter at least one API key")
        
        # Instructions
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        
        with st.expander("🔮 How to get Gemini API Key"):
            st.markdown("""
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Click "Create API key"
            3. Copy the generated key
            4. Paste it in the form above
            """)
        
        with st.expander("🤖 How to get OpenAI API Key"):
            st.markdown("""
            1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Click "Create new secret key"
            3. Copy the generated key
            4. Paste it in the form above
            """)
        
        with st.expander("⚠️ Important Notes"):
            st.markdown("""
            - API keys are stored temporarily for this session only
            - For permanent configuration, set environment variables
            - Keys are required for AI analysis to work
            - Both providers can be used for redundancy
            """)
        
        return False
    
    def render_prompt_editor(self):
        """Render the prompt editor with memory"""
        st.header("📝 Prompt Editor")
        
        # Load last prompt
        if st.button("📥 Load Last Prompt"):
            if st.session_state.last_prompt:
                st.session_state.prompt_text = st.session_state.last_prompt
                # Prompt will show in text area on next interaction
        
        # Default legal analysis prompt
        default_prompt = """Analyze this legal document/message for the inheritance/probate case. Consider:

1. **Key Facts**: Extract relevant factual information
2. **Legal Issues**: Identify potential legal matters (inheritance rights, property disputes, procedural issues)
3. **Evidence Value**: Rate the significance and reliability of this evidence
4. **Timeline Context**: How does this fit into the chronological sequence?
5. **Contradictions**: Note any inconsistencies with previous information
6. **Research Needs**: Identify if additional legal research is required

Context: This is part of a complex inheritance dispute involving properties in Gibraltar and the UK. The deceased is Sean Thweny, and the case involves questions about validity of wills, property transfers, and administrative procedures.

Baseline Reference: {baseline_content}

Document to Analyze:
{document_content}

Provide a concise but thorough analysis with confidence ratings for each finding."""
        
        # Prompt editor
        prompt_text = st.text_area(
            "Analysis Prompt Template",
            value=st.session_state.get('prompt_text', default_prompt),
            height=300,
            help="Use {baseline_content} and {document_content} as placeholders"
        )
        
        # Save prompt
        if st.button("💾 Save Prompt Template"):
            st.session_state.prompt_text = prompt_text
            st.session_state.last_prompt = prompt_text
            self.save_ui_state()
            st.success("Prompt template saved!")
        
        return prompt_text
    
    def render_processing_controls(self):
        """Render comprehensive 5-phase processing controls"""
        st.header("🚀 Multi-Phase Analysis Pipeline")
        
        # Get case summary to show current status
        try:
            case_summary = self.db.get_case_summary()
            phase_status = case_summary.get('phase_status', {})
            next_phase = case_summary.get('next_recommended_phase', 'Unknown')
            
            # Display current status
            st.info(f"**Recommended Next Step:** {next_phase}")
            
            # Phase status overview
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Phase 1", "📂 Ingestion", phase_status.get('phase1_ingestion', '0 documents'))
            with col2:
                st.metric("Phase 2", "🧠 AI Analysis", phase_status.get('phase2_analysis', '0/0'))
            with col3:
                st.metric("Phase 3", "🔍 Deep Research", phase_status.get('phase3_deep_research', '0 sessions'))
            with col4:
                st.metric("Phase 4", "� Timeline", phase_status.get('phase4_timeline', '0 syntheses'))
            with col5:
                st.metric("Phase 5", "📋 Reports", phase_status.get('phase5_reports', '0 reports'))
                
        except Exception as e:
            st.warning(f"Could not load case summary: {e}")
        
        st.markdown("---")
        
        # Phase control buttons
        phase1_col, phase2_col, phase3_col = st.columns(3)
        
        with phase1_col:
            st.subheader("📂 Phase 1: Document Ingestion")
            if st.button("� Ingest Documents", disabled=st.session_state.processing_active, key="phase1_btn"):
                self.start_document_ingestion()
        
        with phase2_col:
            # Use enhanced Phase 2 interface
            self.enhanced_phase2.render_phase2_interface(self)
        
        with phase3_col:
            st.subheader("🔍 Phase 3: Deep Research")
            if st.button("🌐 Cross-Reference", disabled=st.session_state.processing_active, key="phase3_btn"):
                self.start_phase3_deep_research()
        
        # Phases 4 and 5
        phase4_col, phase5_col, control_col = st.columns(3)
        
        with phase4_col:
            st.subheader("📅 Phase 4: Timeline")
            if st.button("⏰ Build Timeline", disabled=st.session_state.processing_active, key="phase4_btn"):
                self.start_phase4_timeline()
        
        with phase5_col:
            st.subheader("📋 Phase 5: Final Report")
            if st.button("📄 Generate Report", disabled=st.session_state.processing_active, key="phase5_btn"):
                self.start_phase5_report()
        
        with control_col:
            st.subheader("⚙️ Controls")
            if st.button("⏹️ Stop Processing", disabled=not st.session_state.processing_active, key="stop_btn"):
                self.stop_processing()
            if st.button("🔄 Refresh Status", key="refresh_btn"):
                st.rerun()
        
        # Show top insights if available
        try:
            case_summary = self.db.get_case_summary()
            top_insights = case_summary.get('top_insights', [])
            if top_insights:
                st.subheader("🎯 Recent High-Confidence Insights")
                for i, insight in enumerate(top_insights[:3]):
                    with st.expander(f"Insight {i+1}: {insight['source']} (Confidence: {insight['confidence']:.2f})"):
                        st.write(insight['summary'])
        except Exception:
            pass
    
    def render_real_time_logs(self):
        """Enhanced real-time logs with detailed API usage tracking"""
        st.header("📋 Real-Time System Logs")
        
        # Create tabs for different log types
        log_tab1, log_tab2, log_tab3, log_tab4 = st.tabs([
            "🔄 Processing Status",
            "🌐 API Usage", 
            "📊 Performance",
            "🔍 Debug Details"
        ])
        
        with log_tab1:
            st.subheader("Current Processing Status")
            
            # Processing state
            processing_active = st.session_state.get('processing_active', False)
            current_phase = st.session_state.get('current_phase', 'idle')
            processed_count = st.session_state.get('processed_count', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                status_color = "🟢" if processing_active else "🔴"
                st.metric("Status", f"{status_color} {'Active' if processing_active else 'Idle'}")
            with col2:
                st.metric("Phase", current_phase.title())
            with col3:
                st.metric("Documents Processed", processed_count)
            
            # Recent documents
            current_doc = st.session_state.get('current_document', 'None')
            st.info(f"📄 Current Document: {current_doc}")
            
            # Progress calculation issues check
            documents_to_process = st.session_state.get('documents_to_process', [])
            total_count = st.session_state.get('total_count', 0)
            if total_count > 0:
                progress_percentage = (processed_count / total_count) * 100
                st.info(f"📊 Progress: {processed_count}/{total_count} ({progress_percentage:.1f}%)")
                if len(documents_to_process) == 0 and processed_count > 0:
                    st.success(f"✅ Processing complete! All {total_count} documents processed.")
            elif processed_count > 0:
                st.warning(f"⚠️ Progress calculation issue: {processed_count} docs processed but total_count = {total_count}")
            
        with log_tab2:
            st.subheader("API Usage Tracking")
            
            # Gemini API usage
            if hasattr(self, 'api_manager') and self.api_manager:
                api_status = self.api_manager.get_api_status()
                st.json(api_status)
                
                # Key rotation info
                st.subheader("🔑 Key Rotation Status")
                current_key = getattr(self.api_manager, 'current_key_index', 'Unknown')
                st.info(f"Current Gemini Key: {current_key}")
            
            # Firecrawl usage tracking
            st.subheader("🌐 Firecrawl Usage")
            if hasattr(self, 'firecrawl_client') and self.firecrawl_client:
                try:
                    # Get Firecrawl status
                    fc_status = {
                        "keys_loaded": len(getattr(self.firecrawl_client, 'api_keys', [])),
                        "current_key_index": getattr(self.firecrawl_client, 'current_key_index', 0),
                        "requests_made": getattr(self.firecrawl_client, 'requests_made', 0),
                        "last_request_time": getattr(self.firecrawl_client, 'last_request_time', 'Never')
                    }
                    st.json(fc_status)
                    
                    # Warning if no requests made
                    if fc_status["requests_made"] == 0:
                        st.warning("⚠️ No Firecrawl requests have been made! This explains the constant processing speed.")
                except Exception as e:
                    st.error(f"Error getting Firecrawl status: {e}")
            
        with log_tab3:
            st.subheader("⚡ Performance Metrics")
            
            # Processing speed
            if processed_count > 0:
                # Calculate processing rate
                start_time = st.session_state.get('processing_start_time')
                if start_time:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    st.metric("Processing Rate", f"{rate:.2f} docs/sec")
                    
                    # This rate should vary if Firecrawl is being used
                    if rate > 10:  # Too fast for AI processing
                        st.warning("⚠️ Processing rate seems too fast for AI analysis. Likely only parsing local files.")
            
            # Database stats
            if hasattr(self, 'db') and self.db:
                try:
                    stats = self.db.get_document_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"Error getting DB stats: {e}")
        
        with log_tab4:
            st.subheader("🔍 Debug Information")
            
            # Session state debugging
            st.subheader("Session State")
            debug_keys = [
                'processing_active', 'current_phase', 'processed_count',
                'current_document', 'documents_to_process'
            ]
            
            debug_state = {}
            for key in debug_keys:
                value = st.session_state.get(key)
                if key == 'documents_to_process' and value:
                    debug_state[key] = f"List with {len(value)} items"
                else:
                    debug_state[key] = str(value)
            
            st.json(debug_state)
            
            # Component status
            st.subheader("Component Status")
            components = {
                'database': hasattr(self, 'db') and self.db is not None,
                'api_manager': hasattr(self, 'api_manager') and self.api_manager is not None,
                'document_parser': hasattr(self, 'doc_parser') and self.doc_parser is not None,
                'firecrawl_client': hasattr(self, 'firecrawl_client') and self.firecrawl_client is not None
            }
            st.json(components)
        
        # Refresh button
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    
    def render_data_viewer(self):
        """Render database content viewer with real-time updates"""
        st.header("🗄️ Data Viewer")
        
        # Add refresh controls at the top
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time data viewer** - Updates automatically during processing")
        with col2:
            auto_refresh = st.checkbox("Auto-refresh", value=True, help="Automatically refresh data every 5 seconds")
        with col3:
            if st.button("🔄 Refresh Now", help="Manually refresh all data"):
                st.rerun()
        
        # Auto-refresh timer
        if auto_refresh and st.session_state.get('processing_active', False):
            time.sleep(1)  # Small delay to prevent too frequent updates
            if 'last_data_refresh' not in st.session_state:
                st.session_state.last_data_refresh = time.time()
            
            # Refresh every 5 seconds during processing
            if time.time() - st.session_state.last_data_refresh > 5:
                st.session_state.last_data_refresh = time.time()
                st.rerun()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Documents", "Analysis Results", "Search", "Live Stats"])
        
        with tab1:
            st.subheader("📄 Documents Database")
            # Document viewer with enhanced accessibility
            docs = self.get_recent_documents()
            if docs:
                # Show document count
                st.info(f"📊 Showing {len(docs)} most recent documents")
                df = pd.DataFrame(docs)
                
                # Make the dataframe always accessible with enhanced styling
                st.dataframe(
                    df, 
                    use_container_width=True,
                    height=400,  # Fixed height for consistency
                    column_config={
                        "confidence_score": st.column_config.ProgressColumn(
                            "Confidence",
                            help="AI confidence score",
                            min_value=0,
                            max_value=1,
                        ),
                        "created_at": st.column_config.DatetimeColumn(
                            "Created",
                            format="MM/DD/YY HH:mm"
                        )
                    }
                )
            else:
                st.info("📄 No documents found. Start by ingesting documents.")
        
        with tab2:
            st.subheader("🧠 Analysis Results")
            # Analysis results viewer with enhanced accessibility
            analyses = self.get_recent_analyses()
            if analyses:
                # Show analysis count and average confidence
                avg_confidence = sum(a.get('confidence_score', 0) for a in analyses) / len(analyses)
                st.info(f"📊 Showing {len(analyses)} analyses | Avg confidence: {avg_confidence:.2f}")
                
                df = pd.DataFrame(analyses)
                
                # Enhanced dataframe with better column configuration
                st.dataframe(
                    df, 
                    use_container_width=True,
                    height=400,
                    column_config={
                        "confidence_score": st.column_config.ProgressColumn(
                            "Confidence",
                            help="Analysis confidence score",
                            min_value=0,
                            max_value=1,
                        ),
                        "relevance_score": st.column_config.ProgressColumn(
                            "Relevance",
                            help="Legal relevance score",
                            min_value=0,
                            max_value=1,
                        ),
                        "created_at": st.column_config.DatetimeColumn(
                            "Created",
                            format="MM/DD/YY HH:mm"
                        )
                    }
                )
                
                # Real-time confidence distribution
                if 'confidence_score' in df.columns and len(df) > 0:
                    st.subheader("📈 Live Confidence Distribution")
                    fig = px.histogram(
                        df, 
                        x='confidence_score', 
                        title="Analysis Confidence Score Distribution",
                        nbins=20,
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_layout(
                        xaxis_title="Confidence Score",
                        yaxis_title="Count",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🧠 No analysis results found. Run Phase 2 analysis to see results.")
        
        with tab3:
            st.subheader("🔍 Search Interface")
            # Enhanced search interface
            search_query = st.text_input("Search documents and analyses:", placeholder="Enter keywords, legal terms, or concepts...")
            
            col1, col2 = st.columns(2)
            with col1:
                search_docs = st.checkbox("Search Documents", value=True)
            with col2:
                search_analyses = st.checkbox("Search Analyses", value=True)
            
            if search_query and st.button("🔍 Search", type="primary"):
                with st.spinner("Searching..."):
                    if search_docs:
                        doc_results = self.db.search_documents(search_query)
                        st.subheader(f"📄 Document Results ({len(doc_results)} found)")
                        if doc_results:
                            for i, result in enumerate(doc_results[:10]):  # Limit to top 10
                                score = result.get('similarity_score', 0)
                                source = result.get('source_file', 'Unknown')
                                with st.expander(f"🎯 Score: {score:.3f} - {Path(source).name}"):
                                    st.write(result.get('content_preview', ''))
                        else:
                            st.info("No document matches found.")
                    
                    if search_analyses:
                        analysis_results = self.db.search_analysis(search_query)
                        st.subheader(f"🧠 Analysis Results ({len(analysis_results)} found)")
                        if analysis_results:
                            for i, result in enumerate(analysis_results[:10]):  # Limit to top 10
                                score = result.get('similarity_score', 0)
                                analysis_type = result.get('analysis_type', 'Unknown')
                                with st.expander(f"🎯 Score: {score:.3f} - {analysis_type}"):
                                    response = result.get('ai_response', '')
                                    st.write(response[:1000] + "..." if len(response) > 1000 else response)
                        else:
                            st.info("No analysis matches found.")
        
        with tab4:
            st.subheader("📊 Live Statistics")
            # Real-time statistics dashboard
            stats = self.db.get_processing_stats()
            
            # Create live metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Documents", stats['total_documents'])
            with metric_col2:
                st.metric("Total Analyses", stats['total_analyses'])
            with metric_col3:
                st.metric("Avg Confidence", f"{stats['average_confidence']}")
            with metric_col4:
                processing_pct = stats['processing_percentage']
                st.metric("Processing Complete", f"{processing_pct}%")
            
            # Progress bar
            st.progress(processing_pct / 100.0)
            
            # Recent activity timeline
            st.subheader("⏱️ Recent Activity")
            recent_docs = self.get_recent_documents(limit=5)
            recent_analyses = self.get_recent_analyses(limit=5)
            
            # Combine and sort by timestamp
            activity = []
            for doc in recent_docs:
                activity.append({
                    'timestamp': doc['created_at'],
                    'type': '📄 Document',
                    'description': f"Added: {Path(doc['source_file']).name}"
                })
            for analysis in recent_analyses:
                activity.append({
                    'timestamp': analysis['created_at'],
                    'type': '🧠 Analysis',
                    'description': f"Analyzed: {analysis['analysis_type']}"
                })
            
            # Sort by timestamp (most recent first)
            activity.sort(key=lambda x: x['timestamp'], reverse=True)
            
            for item in activity[:10]:  # Show last 10 activities
                st.text(f"{item['type']} | {item['timestamp']} | {item['description']}")
    
    def get_recent_documents(self, limit=50):
        """Get recent documents from database with error handling"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, source_file, document_type, content_preview, 
                           character_count, confidence_score, created_at
                    FROM documents 
                    WHERE content IS NOT NULL AND content != ''
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    doc_dict = dict(zip(columns, row))
                    # Ensure confidence_score is a float for display
                    if doc_dict['confidence_score'] is not None:
                        doc_dict['confidence_score'] = float(doc_dict['confidence_score'])
                    else:
                        doc_dict['confidence_score'] = 0.0
                    results.append(doc_dict)
                
                return results
        except Exception as e:
            st.error(f"Error fetching documents: {e}")
            logger.error(f"Database error in get_recent_documents: {e}")
            return []
    
    def get_recent_analyses(self, limit=50):
        """Get recent analyses from database with error handling"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ar.id, ar.document_id, ar.analysis_type, 
                           ar.confidence_score, ar.relevance_score,
                           ar.ai_response, ar.legal_facts, ar.created_at,
                           d.source_file
                    FROM analysis_results ar
                    LEFT JOIN documents d ON ar.document_id = d.id
                    ORDER BY ar.created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    analysis_dict = dict(zip(columns, row))
                    # Ensure scores are floats for display
                    if analysis_dict['confidence_score'] is not None:
                        analysis_dict['confidence_score'] = float(analysis_dict['confidence_score'])
                    else:
                        analysis_dict['confidence_score'] = 0.0
                    
                    if analysis_dict['relevance_score'] is not None:
                        analysis_dict['relevance_score'] = float(analysis_dict['relevance_score'])
                    else:
                        analysis_dict['relevance_score'] = 0.0
                    
                    # Truncate AI response for display
                    if analysis_dict['ai_response'] and len(analysis_dict['ai_response']) > 200:
                        analysis_dict['ai_response_preview'] = analysis_dict['ai_response'][:200] + "..."
                    else:
                        analysis_dict['ai_response_preview'] = analysis_dict['ai_response'] or ""
                    
                    results.append(analysis_dict)
                
                return results
        except Exception as e:
            st.error(f"Error fetching analyses: {e}")
            logger.error(f"Database error in get_recent_analyses: {e}")
            return []
    
    def start_document_ingestion(self):
        """Initialize optimized document ingestion process with caching"""
        logger.info("🚀 Starting optimized document ingestion process...")
        
        # Log the action
        self.log_action("Document Ingestion Started")
        
        # Check if components are initialized
        if not hasattr(self, 'doc_parser') or self.doc_parser is None:
            logger.error("❌ Document parser not initialized")
            st.error("Document parser not initialized. Please refresh the page.")
            return
        
        # Get all documents
        try:
            case_docs_path = st.session_state.current_settings['case_docs_path']
            logger.info(f"📁 Scanning documents in: {case_docs_path}")
            
            documents = self.doc_parser.get_all_documents(case_docs_path)
            logger.info(f"📋 Found {len(documents)} documents to process")
            
            if not documents:
                st.warning("No documents found to process. Check the case docs path.")
                return
            
            # Show document count explanation if high
            if len(documents) > 1000:
                with st.expander("📊 Why so many documents?", expanded=False):
                    explanation = self._explain_document_count(len(documents))
                    st.markdown(explanation)
            
            # Filter cached documents for performance
            new_documents, cached_count = self._filter_cached_documents(documents)
            
            if cached_count > 0:
                st.info(f"📋 Performance optimization: Found {cached_count:,} already processed documents in cache")
                logger.info(f"📋 Cache hit: Skipping {cached_count} documents, processing {len(new_documents)} new ones")
            
            # Initialize ingestion state
            st.session_state.processing_active = True
            st.session_state.current_phase = 'ingestion'
            st.session_state.documents_to_process = documents  # Keep full list for UI
            st.session_state.new_documents_to_process = new_documents  # Actual documents to process
            st.session_state.total_count = len(documents)  # Total for UI display
            st.session_state.new_count = len(new_documents)  # New documents to process
            st.session_state.cached_count = cached_count  # Cached documents
            st.session_state.processed_count = 0
            st.session_state.current_document = ""
            st.session_state.ingestion_batch_size = 25  # Larger batches for better performance
            
            # Show performance summary
            st.success(f"""
            ✅ **Processing Summary:**
            - Total documents found: {len(documents):,}
            - New documents to process: {len(new_documents):,}
            - Cached (skip): {cached_count:,}
            - Cache efficiency: {(cached_count/len(documents)*100):.1f}%
            """)
            
            # Start processing the first batch
            self.continue_document_ingestion()
            
        except Exception as e:
            logger.error(f"❌ Failed to start document ingestion: {e}")
            st.session_state.processing_active = False
            st.error(f"Failed to start ingestion: {e}")
    
    def continue_document_ingestion(self):
        """Continue processing documents with optimized batch processing and caching"""
        if not st.session_state.get('processing_active', False):
            logger.warning("⚠️ Document ingestion called but processing not active")
            return
            
        # Use new documents list if available (filtered for cache)
        documents = st.session_state.get('new_documents_to_process', st.session_state.get('documents_to_process', []))
        if not documents:
            logger.warning("⚠️ No new documents to process")
            self.complete_document_ingestion()
            return
        
        total_docs = st.session_state.get('total_count', len(documents))
        new_docs = len(documents)
        processed_count = st.session_state.get('processed_count', 0)
        cached_count = st.session_state.get('cached_count', 0)
        
        logger.info(f"📦 Optimized processing: {new_docs} new documents, {cached_count} cached, {total_docs} total")
        
        # Create progress indicators
        progress_container = st.container()
        progress_bar = progress_container.progress(processed_count / total_docs)
        status_text = progress_container.empty()
        performance_text = progress_container.empty()
        
        # Track processing start time
        if 'processing_start_time' not in st.session_state:
            st.session_state.processing_start_time = time.time()
        
        # Batch processing for better performance
        batch_size = st.session_state.get('ingestion_batch_size', 25)
        
        # Process documents in batches
        for batch_start in range(processed_count, new_docs, batch_size):
            batch_end = min(batch_start + batch_size, new_docs)
            batch = documents[batch_start:batch_end]
            
            batch_start_time = time.time()
            
            # Update status for batch
            status_text.info(f"� Processing batch {batch_start//batch_size + 1}: documents {batch_start+1}-{batch_end} of {new_docs} new documents")
            
            # Process batch with database optimization
            try:
                doc_ids = []
                for i, doc in enumerate(batch):
                    current_doc_idx = batch_start + i
                    doc_name = doc.get('source_file', 'unknown')
                    
                    # Check if already processed (double check cache)
                    if self._get_document_key(doc) in self.processed_docs:
                        logger.debug(f"📋 Skipping cached document: {doc_name}")
                        continue
                    
                    # Quick content validation
                    content_length = len(str(doc.get('content', '')))
                    if content_length < 10:  # Skip very small documents
                        logger.warning(f"⚠️ Skipping document with insufficient content: {doc_name}")
                        continue
                    
                    # Store document
                    try:
                        doc_start = time.time()
                        doc_id = self.db.store_document(
                            source_file=doc['source_file'],
                            document_type=doc['document_type'],
                            content=doc['content'],
                            timestamp=doc.get('timestamp'),
                            sender=doc.get('sender')
                        )
                        doc_time = time.time() - doc_start
                        
                        # Mark as processed in cache
                        self._mark_document_processed(doc)
                        doc_ids.append(doc_id)
                        
                        # Log only significant events to reduce noise
                        if doc_time > 1.0 or current_doc_idx % 50 == 0:
                            logger.info(f"✅ Processed {doc_name} in {doc_time:.2f}s")
                        
                    except Exception as doc_error:
                        logger.error(f"❌ Failed to store document {doc_name}: {doc_error}")
                        continue
                
                batch_time = time.time() - batch_start_time
                docs_per_sec = len(batch) / batch_time if batch_time > 0 else 0
                
                # Update progress
                st.session_state.processed_count = batch_end
                overall_progress = (batch_end + cached_count) / total_docs
                progress_bar.progress(overall_progress)
                
                # Show performance metrics
                performance_text.info(f"⚡ Batch performance: {len(batch)} docs in {batch_time:.1f}s ({docs_per_sec:.1f} docs/sec)")
                
                # Update processing stats
                self.processing_stats["total_processed"] += len(doc_ids)
                self.processing_stats["processing_time"] += batch_time
                
                logger.info(f"📊 Batch {batch_start//batch_size + 1} complete: {len(doc_ids)} documents stored in {batch_time:.2f}s")
                
                # Shorter delay for better responsiveness
                time.sleep(0.05)  # Much reduced delay
                
            except Exception as batch_error:
                logger.error(f"❌ Batch processing error: {batch_error}")
                continue
        
        # Save cache for development efficiency
        self._save_performance_cache()
        
        # Complete ingestion with performance summary
        total_time = time.time() - st.session_state.processing_start_time
        if new_docs > 0:
            avg_time = total_time / new_docs
        else:
            avg_time = 0
        
        # Final status update
        status_text.success(f"""
        ✅ **Processing Complete!**
        - New documents processed: {new_docs:,}
        - Cached documents skipped: {cached_count:,}
        - Total documents: {total_docs:,}
        - Processing time: {total_time:.2f}s
        - Average time: {avg_time:.3f}s/doc
        - Cache efficiency: {(cached_count/total_docs*100):.1f}%
        """)
        
        logger.info(f"🎉 Optimized ingestion complete: {new_docs} new docs in {total_time:.2f}s (avg: {avg_time:.3f}s/doc), {cached_count} cached")
        
        self.complete_document_ingestion()
    
    def complete_document_ingestion(self):
        """Complete the document ingestion process"""
        st.session_state.processing_active = False
        st.session_state.current_phase = 'completed'  # Set to completed instead of idle
        
        total_processed = st.session_state.get('processed_count', 0)
        success_msg = f"Processed {total_processed} documents"
        logger.info(f"🎉 Document ingestion completed: {success_msg}")
        self.log_action("Document Ingestion Completed", success_msg)
        
        # Clean up ingestion state but preserve counts for progress display
        if 'documents_to_process' in st.session_state:
            del st.session_state.documents_to_process
        # Keep total_count and processed_count for progress display
        
        # Set completion timestamp for auto-reset
        st.session_state.completion_time = time.time()
    
    def start_phase2_analysis(self, limit: Optional[int] = None):
        """Start Phase 2: Individual AI Analysis of documents"""
        st.session_state.processing_active = True
        st.session_state.current_phase = 'phase2_analysis'
        
        self.log_action("Phase 2 Analysis Started", f"Analyzing {'all' if not limit else limit} documents")
        
        try:
            # Initialize analysis engine
            if not hasattr(self, 'analysis_engine'):
                from src.legal_analysis_engine import LegalAnalysisEngine
                self.analysis_engine = LegalAnalysisEngine(self.api_manager)
            
            # Get baseline path
            baseline_path = st.session_state.current_settings.get('baseline_pdf_path')
            
            # Run Phase 2 analysis
            with st.spinner(f"🧠 Running AI analysis on {'all documents' if not limit else f'{limit} documents'}..."):
                results = self.db.analyze_documents_phase2(
                    analysis_engine=self.analysis_engine,
                    firecrawl_client=self.firecrawl_client,
                    baseline_path=baseline_path,
                    limit=limit
                )
            
            # Handle potential None results
            if not results:
                st.error("❌ Phase 2 analysis failed: No results returned")
                return
            
            # Display results
            if results and ('analyzed_documents' in results or 'total_documents' in results):
                st.success(f"✅ Phase 2 Complete: {results.get('analyzed_documents', 0)}/{results.get('total_documents', 0)} documents analyzed")
                st.info(f"🌐 Firecrawl searches: {results.get('firecrawl_searches', 0)}")
                st.info(f"⏱️ Processing time: {results.get('processing_time', 0):.2f}s")
                
                # Show failed documents if any
                if results.get('failed_documents', 0) > 0:
                    st.warning(f"⚠️ {results['failed_documents']} documents failed analysis")
                
                # Show analysis details if available
                if 'analysis_details' in results and results['analysis_details']:
                    with st.expander("📊 Analysis Details"):
                        for detail in results['analysis_details'][:5]:  # Show first 5
                            st.write(f"**{detail['source_file']}**: Confidence {detail['confidence_score']:.2f}, Evidence: {detail['evidence_value']}")
            else:
                st.error(f"❌ Phase 2 failed: {results.get('error', 'Unknown error - no results returned')}")
                
        except Exception as e:
            st.error(f"❌ Phase 2 analysis failed: {e}")
            logger.error(f"Phase 2 analysis error: {e}")
        finally:
            st.session_state.processing_active = False
            st.session_state.current_phase = 'idle'

    def start_phase2_analysis_enhanced(self, limit: Optional[int] = None, debug_mode: bool = False, debugger=None):
        """Enhanced Phase 2 analysis with real-time debugging and comprehensive error handling"""
        
        if debugger and debug_mode:
            debugger.log_debug(f"🚀 Starting enhanced Phase 2 analysis (limit: {limit})", "DEBUG")
        
        st.session_state.processing_active = True
        st.session_state.current_phase = 'phase2_analysis'
        
        # Create real-time progress containers
        progress_container = st.container()
        metrics_container = st.container()
        log_container = st.container()
        
        start_time = time.time()
        analyzed_count = 0
        failed_count = 0
        
        try:
            # Initialize analysis engine with validation
            if not hasattr(self, 'analysis_engine'):
                if debugger and debug_mode:
                    debugger.log_debug("Initializing analysis engine...", "DEBUG")
                from src.legal_analysis_engine import LegalAnalysisEngine
                self.analysis_engine = LegalAnalysisEngine(self.api_manager)
                if debugger and debug_mode:
                    debugger.log_debug("✅ Analysis engine initialized", "SUCCESS")
            
            # Validate API keys
            if debugger and debug_mode:
                debugger.log_debug("Validating API connectivity...", "DEBUG")
            
            api_status = self.api_manager.get_api_status()
            working_keys = api_status.get('working_keys', 0)
            
            if working_keys == 0:
                error_msg = f"No working Gemini 2.5 Pro API keys available ({working_keys}/{api_status.get('total_keys', 0)})"
                if debugger:
                    debugger.log_debug(f"❌ {error_msg}", "ERROR")
                st.error(f"❌ {error_msg}")
                st.error("The system requires working Gemini 2.5 Pro API keys to function")
                return
            
            if debugger and debug_mode:
                debugger.log_debug(f"✅ {working_keys} API providers available", "SUCCESS")
            
            # Get baseline path
            baseline_path = st.session_state.current_settings.get('baseline_pdf_path')
            if debugger and debug_mode:
                debugger.log_debug(f"Baseline path: {baseline_path}", "DEBUG")
            
            # Get documents for analysis with validation
            if debugger and debug_mode:
                debugger.log_debug("Retrieving documents for analysis...", "DEBUG")
            
            documents = self.db.get_documents_for_analysis(limit=limit)
            
            if not documents:
                error_msg = "No documents available for analysis"
                if debugger:
                    debugger.log_debug(f"❌ {error_msg}", "ERROR")
                st.error(f"❌ {error_msg}")
                return
            
            total_docs = len(documents)
            if debugger and debug_mode:
                debugger.log_debug(f"✅ Found {total_docs} documents for analysis", "SUCCESS")
            
            # Create progress tracking
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
            
            # Process each document with real-time feedback
            for i, doc in enumerate(documents):
                doc_start_time = time.time()
                current_progress = (i + 1) / total_docs
                
                # Update progress display
                progress_bar.progress(current_progress)
                doc_source = doc.get('source_file', 'Unknown')
                doc_filename = Path(doc_source).name if doc_source else 'Unknown'
                status_text.text(f"📄 Processing {i+1}/{total_docs}: {doc_filename}")
                
                elapsed_time = time.time() - start_time
                time_text.text(f"⏱️ Elapsed: {elapsed_time:.1f}s")
                
                if debugger and debug_mode:
                    debugger.log_debug(f"Processing document {i+1}/{total_docs}: {doc_filename}", "DEBUG")
                
                # Validate document content
                content = doc.get('content', '')
                content_length = len(content) if content else 0
                doc_id = doc.get('id')
                
                if not content or content_length < 20:
                    error_msg = f"Document {doc_filename} has insufficient content ({content_length} chars)"
                    if debugger:
                        debugger.log_debug(f"⚠️ {error_msg}", "WARNING")
                    failed_count += 1
                    
                    # Mark as failed in database
                    try:
                        self.db._update_document_analysis_failed(doc_id, "Insufficient content")
                    except:
                        pass  # Continue even if update fails
                    continue
                
                try:
                    # Perform the actual analysis
                    if debugger and debug_mode:
                        debugger.log_debug(f"Analyzing content ({content_length} chars)...", "DEBUG")
                    
                    analysis_result = self.analysis_engine.analyze_document(
                        content, 
                        "comprehensive", 
                        None  # Baseline content can be added later
                    )
                    
                    # Store analysis result
                    analysis_id = self.db.store_analysis_result(
                        document_id=doc_id,
                        analysis_type=analysis_result.analysis_type,
                        prompt_used=f"Enhanced Phase 2 Analysis",
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
                            "evidence_value": analysis_result.evidence_value
                        }),
                        processing_time=analysis_result.processing_time
                    )
                    
                    # Mark document as analyzed
                    self.db._update_document_analysis_complete(doc_id, analysis_result.confidence_score)
                    
                    analyzed_count += 1
                    doc_time = time.time() - doc_start_time
                    
                    if debugger and debug_mode:
                        debugger.log_debug(f"✅ Analysis complete: {doc_filename} (confidence: {analysis_result.confidence_score:.2f}, {doc_time:.2f}s)", "SUCCESS")
                    
                except Exception as e:
                    failed_count += 1
                    doc_time = time.time() - doc_start_time
                    error_msg = f"Analysis failed for {doc_filename}: {str(e)}"
                    
                    if debugger:
                        debugger.log_debug(f"❌ {error_msg}", "ERROR")
                    
                    # Mark as failed in database
                    try:
                        self.db._update_document_analysis_failed(doc_id, str(e))
                    except:
                        pass
                
                # Update real-time metrics
                with metrics_container:
                    st.write(f"✅ **Analyzed**: {analyzed_count} | ❌ **Failed**: {failed_count} | 📊 **Progress**: {current_progress:.1%} | ⏱️ **Time**: {time.time() - start_time:.1f}s")
            
            # Final results
            total_time = time.time() - start_time
            
            if debugger and debug_mode:
                debugger.log_debug(f"🎉 Analysis complete: {analyzed_count} analyzed, {failed_count} failed in {total_time:.2f}s", "SUCCESS")
            
            # Display final summary
            if analyzed_count > 0:
                st.success(f"✅ Phase 2 Complete: {analyzed_count}/{total_docs} documents analyzed in {total_time:.2f}s")
            
            if failed_count > 0:
                st.warning(f"⚠️ {failed_count} documents failed analysis")
                if debugger and debug_mode:
                    st.info("Check debug logs above for detailed failure reasons")
            
            # Show performance metrics
            if total_docs > 0:
                avg_time = total_time / total_docs
                st.info(f"📊 Average processing time: {avg_time:.2f}s per document")
            
        except Exception as e:
            error_msg = f"Phase 2 analysis failed: {e}"
            if debugger:
                debugger.log_debug(f"❌ {error_msg}", "ERROR")
            logger.error(f"Enhanced Phase 2 analysis error: {e}")
            st.error(f"❌ {error_msg}")
            
            # Show full traceback in debug mode
            if debug_mode:
                st.code(traceback.format_exc())
        
        finally:
            st.session_state.processing_active = False
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete")

    def start_phase3_deep_research(self):
        """Start Phase 3: Deep Research & Cross-Referencing"""
        st.session_state.processing_active = True
        st.session_state.current_phase = 'phase3_deep_research'
        
        self.log_action("Phase 3 Deep Research Started")
        
        try:
            # Initialize analysis engine if needed
            if not hasattr(self, 'analysis_engine'):
                from src.legal_analysis_engine import LegalAnalysisEngine
                self.analysis_engine = LegalAnalysisEngine(self.api_manager)
            
            baseline_path = st.session_state.current_settings.get('baseline_pdf_path')
            
            with st.spinner("🔍 Running deep research and cross-referencing..."):
                results = self.db.run_deep_research_phase3(
                    analysis_engine=self.analysis_engine,
                    baseline_path=baseline_path
                )
            
            if results['status'] == 'success':
                st.success("✅ Phase 3 Complete: Deep research and cross-referencing finished")
                
                with st.expander("🔍 Deep Research Summary"):
                    st.write(results['summary'])
                
                if results.get('contradictions'):
                    with st.expander("⚠️ Contradictions Found"):
                        for contradiction in results['contradictions']:
                            st.write(f"• {contradiction}")
            else:
                st.error(f"❌ Phase 3 failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Phase 3 deep research failed: {e}")
            logger.error(f"Phase 3 error: {e}")
        finally:
            st.session_state.processing_active = False
            st.session_state.current_phase = 'idle'

    def start_phase4_timeline(self):
        """Start Phase 4: Timeline & Fact Synthesis"""
        st.session_state.processing_active = True
        st.session_state.current_phase = 'phase4_timeline'
        
        self.log_action("Phase 4 Timeline Synthesis Started")
        
        try:
            if not hasattr(self, 'analysis_engine'):
                from src.legal_analysis_engine import LegalAnalysisEngine
                self.analysis_engine = LegalAnalysisEngine(self.api_manager)
            
            baseline_path = st.session_state.current_settings.get('baseline_pdf_path')
            
            with st.spinner("📅 Building timeline and synthesizing facts..."):
                results = self.db.build_timeline_phase4(
                    analysis_engine=self.analysis_engine,
                    baseline_path=baseline_path
                )
            
            if results['status'] == 'success':
                st.success("✅ Phase 4 Complete: Timeline and fact synthesis finished")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Timeline Events", results['timeline_events'])
                with col2:
                    st.metric("Key Actors", results['key_actors'])
                with col3:
                    st.metric("Legal Deadlines", results['legal_deadlines'])
                
                with st.expander("📅 Timeline Synthesis"):
                    st.write(results['synthesis'])
                
                if results.get('contradictions'):
                    with st.expander("⚠️ Timeline Contradictions"):
                        for contradiction in results['contradictions']:
                            st.write(f"• {contradiction}")
            else:
                st.error(f"❌ Phase 4 failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Phase 4 timeline synthesis failed: {e}")
            logger.error(f"Phase 4 error: {e}")
        finally:
            st.session_state.processing_active = False
            st.session_state.current_phase = 'idle'

    def start_phase5_report(self):
        """Start Phase 5: Comprehensive Report Generation"""
        st.session_state.processing_active = True
        st.session_state.current_phase = 'phase5_report'
        
        self.log_action("Phase 5 Report Generation Started")
        
        try:
            if not hasattr(self, 'analysis_engine'):
                from src.legal_analysis_engine import LegalAnalysisEngine
                self.analysis_engine = LegalAnalysisEngine(self.api_manager)
            
            baseline_path = st.session_state.current_settings.get('baseline_pdf_path')
            
            with st.spinner("📋 Generating comprehensive legal report..."):
                results = self.db.generate_comprehensive_report_phase5(
                    analysis_engine=self.analysis_engine,
                    baseline_path=baseline_path,
                    report_type="comprehensive"
                )
            
            if results['status'] == 'success':
                st.success("✅ Phase 5 Complete: Comprehensive report generated")
                
                # Display case statistics
                stats = results['case_statistics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Documents", stats['total_documents'])
                with col2:
                    st.metric("Analyses", stats['total_analyses'])
                with col3:
                    st.metric("Avg Confidence", f"{stats['average_analysis_confidence']:.2f}")
                with col4:
                    st.metric("Research Sessions", stats['deep_research_sessions'])
                
                # Show report content
                with st.expander("📋 Comprehensive Legal Report", expanded=True):
                    st.markdown(results['report'])
                
                # Research recommendations
                if results.get('research_recommendations'):
                    with st.expander("🔍 Research Recommendations"):
                        for rec in results['research_recommendations']:
                            st.write(f"• {rec}")
                
                # Download option
                st.download_button(
                    label="💾 Download Report",
                    data=results['report'],
                    file_name=f"legal_case_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            else:
                st.error(f"❌ Phase 5 failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Phase 5 report generation failed: {e}")
            logger.error(f"Phase 5 error: {e}")
        finally:
            st.session_state.processing_active = False
            st.session_state.current_phase = 'idle'

    def start_individual_analysis(self):
        """Start Phase 2: Individual AI Analysis of stored documents"""
        logger.info("🧠 Starting Phase 2: Individual AI Analysis...")
        
        # Log the action
        self.log_action("Individual AI Analysis Started")
        
        # Check if components are initialized
        if not hasattr(self, 'analysis_engine') or self.analysis_engine is None:
            logger.error("❌ Analysis engine not initialized")
            st.error("Analysis engine not initialized. Please refresh the page.")
            return
        
        if not hasattr(self, 'db') or self.db is None:
            logger.error("❌ Database not initialized")
            st.error("Database not initialized. Please refresh the page.")
            return
        
        try:
            # Get all unanalyzed documents from database
            logger.info("📋 Fetching documents for analysis...")
            unanalyzed_docs = self.db.get_unanalyzed_documents()
            
            if not unanalyzed_docs:
                st.warning("No unanalyzed documents found. Please run document ingestion first.")
                return
            
            logger.info(f"📊 Found {len(unanalyzed_docs)} documents to analyze")
            
            # Initialize analysis state
            st.session_state.processing_active = True
            st.session_state.current_phase = 'analysis'
            st.session_state.documents_to_analyze = unanalyzed_docs
            st.session_state.total_count = len(unanalyzed_docs)
            st.session_state.processed_count = 0
            st.session_state.current_document = ""
            st.session_state.analysis_start_time = time.time()
            
            # Get baseline content for context
            baseline_content = st.session_state.get('baseline_content', '')
            if not baseline_content:
                logger.warning("⚠️ No baseline content available for analysis context")
            
            # Start processing the documents
            self.continue_individual_analysis()
            
        except Exception as e:
            logger.error(f"❌ Failed to start individual analysis: {e}")
            st.session_state.processing_active = False
            st.error(f"Failed to start analysis: {e}")
    
    def continue_individual_analysis(self):
        """Continue processing documents with AI analysis"""
        if not st.session_state.get('processing_active', False):
            logger.warning("⚠️ Individual analysis called but processing not active")
            return
            
        if not st.session_state.get('documents_to_analyze'):
            logger.warning("⚠️ No documents to analyze in continue_individual_analysis")
            self.complete_individual_analysis()
            return
        
        documents = st.session_state.documents_to_analyze
        processed_count = st.session_state.get('processed_count', 0)
        total_docs = len(documents)
        baseline_content = st.session_state.get('baseline_content', '')
        
        logger.info(f"🧠 Starting AI analysis: {total_docs} documents, {processed_count} already processed")
        
        # Create progress indicators
        progress_container = st.container()
        progress_bar = progress_container.progress(processed_count / total_docs)
        status_text = progress_container.empty()
        
        # Process remaining documents
        for current_idx in range(processed_count, min(processed_count + 3, total_docs)):  # Process 3 at a time
            doc = documents[current_idx]
            doc_id = doc['id']
            doc_content = doc['content']
            source_file = doc['source_file']
            
            # Update status
            status_msg = f"🧠 Analyzing document {current_idx+1}/{total_docs}: {source_file}"
            status_text.info(status_msg)
            logger.info(status_msg)
            
            try:
                # Perform AI analysis
                analysis_start = time.time()
                logger.info(f"🔍 Starting Gemini analysis for {doc_id}")
                
                # Run analysis with baseline context
                analysis_result = self.analysis_engine.analyze_document(
                    document_content=doc_content,
                    analysis_type="comprehensive",
                    baseline_content=baseline_content
                )
                
                analysis_time = time.time() - analysis_start
                logger.info(f"✅ Analysis completed in {analysis_time:.2f}s - Relevance: {analysis_result.relevance_score:.2f}, Confidence: {analysis_result.confidence_score:.2f}")
                
                # Store analysis results in database
                analysis_id = self.db.store_analysis_result(
                    document_id=doc_id,
                    analysis_type=analysis_result.analysis_type,
                    prompt_used=f"Comprehensive legal analysis with baseline context",
                    ai_response=analysis_result.summary,
                    confidence_score=analysis_result.confidence_score,
                    relevance_score=analysis_result.relevance_score,
                    legal_facts=analysis_result.key_facts,
                    legal_laws=analysis_result.legal_issues,
                    firecrawl_triggered=analysis_result.needs_legal_research,
                    firecrawl_results=self._format_analysis_metadata(analysis_result),
                    processing_time=analysis_result.processing_time
                )
                
                # Update document status
                self.db.mark_document_analyzed(doc_id, analysis_result.confidence_score)
                
                # Check if Firecrawl research is needed
                if analysis_result.needs_legal_research and st.session_state.current_settings.get('enable_firecrawl', False):
                    logger.info(f"🌐 Triggering Firecrawl research for {doc_id}")
                    self._trigger_firecrawl_research(doc_id, analysis_result)
                
                logger.info(f"💾 Analysis stored with ID: {analysis_id}")
                
            except Exception as doc_error:
                logger.error(f"❌ Failed to analyze document {doc_id}: {doc_error}")
                # Continue with next document instead of failing completely
                continue
            
            # Update progress
            st.session_state.processed_count = current_idx + 1
            st.session_state.current_document = source_file
            progress_bar.progress((current_idx + 1) / total_docs)
            
            # Trigger real-time UI updates
            self.refresh_header_metrics()
            
            # Brief delay for UI updates
            time.sleep(0.5)
        
        # Check if we need to continue or complete
        if st.session_state.processed_count < total_docs:
            # Continue processing in next iteration
            logger.info(f"📊 Processed {st.session_state.processed_count}/{total_docs} - continuing...")
            st.rerun()
        else:
            # Complete analysis
            total_time = time.time() - st.session_state.analysis_start_time
            avg_time = total_time / total_docs
            status_text.success(f"✅ Completed AI analysis of {total_docs} documents in {total_time:.2f}s (avg: {avg_time:.2f}s/doc)")
            logger.info(f"🎉 Individual AI analysis completed: {total_docs} documents in {total_time:.2f}s")
            
            self.complete_individual_analysis()
    
    def _format_analysis_metadata(self, analysis_result) -> str:
        """Format analysis metadata as JSON string"""
        metadata = {
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
        }
        return json.dumps(metadata)
    
    def _trigger_firecrawl_research(self, doc_id: str, analysis_result):
        """Trigger Firecrawl research for legal topics"""
        try:
            logger.info(f"🌐 Starting Firecrawl research for document {doc_id}")
            
            # Research each legal topic
            for topic in analysis_result.research_topics[:3]:  # Limit to 3 topics to avoid rate limits
                logger.info(f"🔍 Researching: {topic}")
                
                # Use Firecrawl to search for legal information
                search_query = f"{topic} inheritance law probate Gibraltar UK"
                research_results = self.firecrawl_client.search_legal_topic(search_query)
                
                if research_results:
                    # Store research results
                    self.db.store_research_result(doc_id, topic, research_results)
                    logger.info(f"✅ Research completed for: {topic}")
                else:
                    logger.warning(f"⚠️ No research results for: {topic}")
        
        except Exception as e:
            logger.error(f"❌ Firecrawl research failed: {e}")
    
    def complete_individual_analysis(self):
        """Complete the individual analysis process"""
        st.session_state.processing_active = False
        st.session_state.current_phase = 'analysis_completed'
        
        total_processed = st.session_state.get('processed_count', 0)
        success_msg = f"Analyzed {total_processed} documents with AI"
        logger.info(f"🎉 Individual analysis completed: {success_msg}")
        self.log_action("Individual AI Analysis Completed", success_msg)
        
        # Clean up analysis state
        if 'documents_to_analyze' in st.session_state:
            del st.session_state.documents_to_analyze
        if 'analysis_start_time' in st.session_state:
            del st.session_state.analysis_start_time
            
        # Set completion timestamp
        st.session_state.analysis_completion_time = time.time()
    
    def start_deep_research(self):
        """Start deep research phase"""
        st.session_state.processing_active = True
        st.session_state.current_phase = 'deep_research'
        self.log_action("Deep Research Started")
        
        st.info("Deep research started - this would perform cross-referencing and Firecrawl searches")
    
    def start_test_analysis(self, count):
        """Start test analysis with limited documents"""
        self.log_action("Test Analysis Started", f"Testing with {count} documents")
        st.info(f"Test analysis started with {count} documents")
    
    def stop_processing(self):
        """Stop current processing"""
        st.session_state.processing_active = False
        st.session_state.current_phase = 'idle'
        self.log_action("Processing Stopped", "User requested stop")
        st.warning("Processing stopped by user")
    
    def log_action(self, action, details=""):
        """Log an action to session history"""
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': action,
            'details': details
        }
        st.session_state.session_history.append(entry)
        
        # Keep only last 100 entries
        if len(st.session_state.session_history) > 100:
            st.session_state.session_history = st.session_state.session_history[-100:]
        
        # Auto-save
        self.save_ui_state()
    
    def clear_progress_state(self):
        """Clear the processing progress state"""
        st.session_state.processing_active = False
        st.session_state.current_phase = 'idle'
        st.session_state.processed_count = 0
        st.session_state.total_count = 0
        st.session_state.current_document = None
        
        # Clear completion timestamp
        if 'completion_time' in st.session_state:
            del st.session_state.completion_time
        
        # Clear any remaining processing state
        if 'documents_to_process' in st.session_state:
            del st.session_state.documents_to_process
        if 'processing_start_time' in st.session_state:
            del st.session_state.processing_start_time
            
        self.log_action("Progress Cleared", "User cleared processing progress")
        logger.info("🗑️ Progress state cleared by user")

    def show_ingestion_progress(self):
        """Show real-time ingestion progress without page refreshes"""
        processed = st.session_state.get('processed_count', 0)
        total_count = st.session_state.get('total_count', 0)
        current_doc = st.session_state.get('current_document', '')
        processing_active = st.session_state.get('processing_active', False)
        current_phase = st.session_state.get('current_phase', 'idle')
        
        # Always show progress if we have processed documents or are processing
        if total_count > 0:
            progress = processed / total_count
            st.progress(progress)
            
            # Show status based on whether processing is active or completed
            if processing_active and current_phase == 'ingestion':
                st.info(f"📄 Processing: {current_doc} ({processed}/{total_count})")
            elif current_phase == 'completed':
                # Show completion message with action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"✅ Completed: {processed}/{total_count} documents processed")
                with col2:
                    if st.button("🔍 Start Analysis", key="post_ingest_analysis"):
                        self.start_individual_analysis()
                        st.rerun()
                with col3:
                    if st.button("🗑️ Clear Progress", key="clear_progress"):
                        self.clear_progress_state()
                        st.rerun()
            else:
                # Processing stopped or other state
                if processed >= total_count:
                    st.success(f"✅ Completed: {processed}/{total_count} documents processed")
                else:
                    st.warning(f"⚠️ Stopped: {processed}/{total_count} documents processed")
            
            # Show recent activity
            if hasattr(st.session_state, 'recent_activities'):
                with st.expander("Recent Activity", expanded=False):
                    for activity in st.session_state.recent_activities[-5:]:  # Show last 5
                        st.text(activity)
        else:
            # No documents to process
            if processing_active:
                st.info("📄 Initializing document processing...")
            else:
                st.info("📄 No documents processed yet")

    def run(self):
        """Main application entry point with real-time updates"""
        self.render_header()
        self.render_sidebar()
        
        # Show ingestion progress banner if active
        if st.session_state.get('processing_active', False) and st.session_state.get('current_phase') == 'ingestion':
            with st.container():
                st.info("📦 Document Ingestion in Progress...")
                self.show_ingestion_progress()
            
            # Continue ingestion if documents are still being processed
            if st.session_state.get('documents_to_process'):
                logger.info("🔄 Continuing document ingestion...")
                self.continue_document_ingestion()
                # DON'T return here - let users access tabs during processing
        
        # Real-time metric updates during processing
        if st.session_state.get('processing_active', False):
            # Refresh header metrics in real-time
            self.refresh_header_metrics()
            
            # Auto-refresh every 3 seconds during processing
            if 'last_ui_refresh' not in st.session_state:
                st.session_state.last_ui_refresh = time.time()
            
            if time.time() - st.session_state.last_ui_refresh > 3:
                st.session_state.last_ui_refresh = time.time()
                st.rerun()
        
        # Main content area - ALWAYS accessible and fully visible
        main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
            "🎛️ Control Panel", 
            "📝 Prompt Editor", 
            "📊 Data Viewer", 
            "📋 Logs"
        ])
        
        with main_tab1:
            self.render_processing_controls()
        
        with main_tab2:
            self.render_prompt_editor()
        
        with main_tab3:
            # Ensure data viewer is always fully accessible
            self.render_data_viewer()
        
        with main_tab4:
            self.render_real_time_logs()
        
        # Auto-save state periodically
        if 'last_save' not in st.session_state:
            st.session_state.last_save = time.time()
        
        if time.time() - st.session_state.last_save > 30:  # Save every 30 seconds
            self.save_ui_state()
            st.session_state.last_save = time.time()

# Main execution
if __name__ == "__main__":
    app = LegalCaseAnalysisUI()
    app.run()
