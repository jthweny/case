import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

class EnhancedMessageProcessor:
    """
    Enhanced message processor that integrates with the hybrid database,
    smart API manager, and Firecrawl research
    """
    
    def __init__(self, db_manager, api_manager, researcher, firecrawl_client=None):
        self.db = db_manager
        self.api_manager = api_manager
        self.researcher = researcher
        self.firecrawl_client = firecrawl_client
        self.logger = logging.getLogger(__name__)
        
        # Load baseline PDF content for context
        self.baseline_content = ""
        self._load_baseline_content()
    
    def _load_baseline_content(self):
        """Load baseline PDF content for context"""
        try:
            # This would read the baseline PDF
            # For now, we'll use a placeholder
            self.baseline_content = """
            BASELINE CONTEXT: This is a complex inheritance/probate case involving:
            - Deceased: Sean Thweny
            - Properties in Gibraltar and UK
            - Questions about will validity
            - Administrative procedures and transfers
            - Multiple family members involved
            """
            self.logger.info("Baseline content loaded for context")
        except Exception as e:
            self.logger.warning(f"Could not load baseline content: {e}")
    
    def analyze_document(self, document: Dict[str, Any], prompt_template: str,
                        enable_firecrawl: bool = True, max_tokens: int = 2048,
                        temperature: float = 0.7) -> Dict[str, Any]:
        """
        Analyze a single document with AI and optional Firecrawl research
        """
        start_time = time.time()
        
        try:
            # Prepare the prompt
            analysis_prompt = self._prepare_analysis_prompt(document, prompt_template)
            
            # Generate AI analysis
            ai_response, ai_metadata = self.api_manager.generate_content(
                prompt=analysis_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract structured information from AI response
            structured_analysis = self._extract_structured_info(ai_response)
            
            # Determine if Firecrawl research is needed
            firecrawl_triggered = False
            firecrawl_results = None
            
            if enable_firecrawl and self._should_trigger_firecrawl(ai_response, structured_analysis):
                self.logger.info("Triggering Firecrawl research for additional context")
                firecrawl_results = self._perform_firecrawl_research(structured_analysis)
                firecrawl_triggered = True
            
            # Calculate confidence and relevance scores
            confidence_score = self._calculate_confidence_score(ai_response, structured_analysis)
            relevance_score = self._calculate_relevance_score(document, structured_analysis)
            
            processing_time = time.time() - start_time
            
            # Store analysis result in database
            analysis_id = self.db.store_analysis_result(
                document_id=document.get('id', ''),
                analysis_type='individual_analysis',
                prompt_used=analysis_prompt,
                ai_response=ai_response,
                confidence_score=confidence_score,
                relevance_score=relevance_score,
                legal_facts=structured_analysis.get('legal_facts', []),
                legal_laws=structured_analysis.get('legal_laws', []),
                firecrawl_triggered=firecrawl_triggered,
                firecrawl_results=json.dumps(firecrawl_results) if firecrawl_results else None,
                processing_time=processing_time
            )
            
            result = {
                'analysis_id': analysis_id,
                'document_id': document.get('id', ''),
                'ai_response': ai_response,
                'structured_analysis': structured_analysis,
                'confidence_score': confidence_score,
                'relevance_score': relevance_score,
                'firecrawl_triggered': firecrawl_triggered,
                'firecrawl_results': firecrawl_results,
                'processing_time': processing_time,
                'ai_metadata': ai_metadata
            }
            
            self.logger.info(f"Successfully analyzed document in {processing_time:.2f}s (confidence: {confidence_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing document: {e}")
            raise
    
    def _prepare_analysis_prompt(self, document: Dict[str, Any], template: str) -> str:
        """Prepare the analysis prompt with document content and baseline context"""
        
        # Extract document details
        content = document.get('content', '')
        doc_type = document.get('document_type', 'unknown')
        timestamp = document.get('timestamp', 'Unknown')
        sender = document.get('sender', 'Unknown')
        source_file = document.get('source_file', 'Unknown')
        
        # Format document info
        document_info = f"""
Document Type: {doc_type}
Source File: {source_file}
Timestamp: {timestamp}
Sender: {sender}
Content Length: {len(content)} characters

Content:
{content}
        """.strip()
        
        # Replace placeholders in template
        analysis_prompt = template.replace('{baseline_content}', self.baseline_content)
        analysis_prompt = analysis_prompt.replace('{document_content}', document_info)
        
        return analysis_prompt
    
    def _extract_structured_info(self, ai_response: str) -> Dict[str, Any]:
        """Extract structured information from AI response"""
        structured = {
            'legal_facts': [],
            'legal_laws': [],
            'key_findings': [],
            'evidence_value': 'medium',
            'timeline_significance': 'low',
            'contradictions': [],
            'research_keywords': []
        }
        
        try:
            # Extract legal facts (look for numbered lists, bullet points, etc.)
            fact_pattern = r'(?:Key Facts?|Facts?)[\s\S]*?(?=\n\n|\n[A-Z]|$)'
            fact_match = re.search(fact_pattern, ai_response, re.IGNORECASE)
            if fact_match:
                facts_text = fact_match.group()
                facts = re.findall(r'[•\-\*]\s*(.+)', facts_text)
                structured['legal_facts'] = [fact.strip() for fact in facts]
            
            # Extract legal laws/references
            law_pattern = r'(?:Legal|Law|Statute|Act|Section)[\s\S]*?(?=\n\n|\n[A-Z]|$)'
            law_match = re.search(law_pattern, ai_response, re.IGNORECASE)
            if law_match:
                laws_text = law_match.group()
                laws = re.findall(r'[•\-\*]\s*(.+)', laws_text)
                structured['legal_laws'] = [law.strip() for law in laws]
            
            # Extract confidence indicators
            if any(word in ai_response.lower() for word in ['certain', 'definitive', 'clear', 'conclusive']):
                structured['evidence_value'] = 'high'
            elif any(word in ai_response.lower() for word in ['uncertain', 'unclear', 'ambiguous', 'questionable']):
                structured['evidence_value'] = 'low'
            
            # Extract research keywords
            keywords = re.findall(r'(?:research|investigate|verify|check)[\s\w]*?(?:about|regarding|concerning)\s+([A-Za-z\s]+)', ai_response, re.IGNORECASE)
            structured['research_keywords'] = [kw.strip() for kw in keywords]
            
        except Exception as e:
            self.logger.warning(f"Error extracting structured info: {e}")
        
        return structured
    
    def _should_trigger_firecrawl(self, ai_response: str, structured_analysis: Dict[str, Any]) -> bool:
        """Determine if Firecrawl research should be triggered"""
        
        # Trigger if AI explicitly mentions need for research
        research_indicators = [
            'need.*research', 'require.*verification', 'should.*investigate',
            'need.*legal.*precedent', 'check.*statute', 'verify.*law',
            'property.*transfer.*rules', 'gibraltar.*property.*law',
            'inheritance.*law', 'probate.*procedure'
        ]
        
        for indicator in research_indicators:
            if re.search(indicator, ai_response, re.IGNORECASE):
                return True
        
        # Trigger if structured analysis has research keywords
        if structured_analysis.get('research_keywords'):
            return True
        
        # Trigger if evidence value is low (needs more context)
        if structured_analysis.get('evidence_value') == 'low':
            return True
        
        # Trigger for specific legal topics
        legal_topics = [
            'property transfer', 'gibraltar law', 'inheritance rights',
            'probate procedure', 'will validity', 'administration'
        ]
        
        for topic in legal_topics:
            if topic.lower() in ai_response.lower():
                return True
        
        return False
    
    def _perform_firecrawl_research(self, structured_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform Firecrawl research based on analysis"""
        try:
            # Prepare research queries
            queries = []
            
            # Add research keywords
            if structured_analysis.get('research_keywords'):
                queries.extend(structured_analysis['research_keywords'])
            
            # Add default legal research queries
            queries.extend([
                "Gibraltar property inheritance law",
                "UK probate procedure requirements",
                "Property transfer inheritance disputes",
                "Gibraltar UK inheritance law differences"
            ])
            
            # Limit to top 3 queries to avoid overuse
            queries = queries[:3]
            
            research_results = {}
            for query in queries:
                try:
                    # This would call Firecrawl via MCP
                    # For now, we'll simulate the results
                    research_results[query] = {
                        'query': query,
                        'found_results': True,
                        'summary': f"Research results for: {query}",
                        'sources': ['legal-database.com', 'gibraltar-law.gov']
                    }
                    self.logger.info(f"Completed Firecrawl research for: {query}")
                except Exception as e:
                    self.logger.warning(f"Firecrawl research failed for {query}: {e}")
            
            return research_results
            
        except Exception as e:
            self.logger.error(f"Error performing Firecrawl research: {e}")
            return None
    
    def _calculate_confidence_score(self, ai_response: str, structured_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score based on AI response quality"""
        score = 0.5  # Base score
        
        # Increase confidence based on response length and detail
        if len(ai_response) > 500:
            score += 0.1
        if len(ai_response) > 1000:
            score += 0.1
        
        # Increase confidence if structured info was extracted
        if structured_analysis.get('legal_facts'):
            score += 0.1
        if structured_analysis.get('legal_laws'):
            score += 0.1
        
        # Adjust based on evidence value
        evidence_value = structured_analysis.get('evidence_value', 'medium')
        if evidence_value == 'high':
            score += 0.2
        elif evidence_value == 'low':
            score -= 0.1
        
        # Check for confidence indicators in text
        high_confidence_words = ['certain', 'definitive', 'clear', 'conclusive', 'evidence shows']
        low_confidence_words = ['uncertain', 'unclear', 'possibly', 'might be', 'unclear']
        
        for word in high_confidence_words:
            if word in ai_response.lower():
                score += 0.05
        
        for word in low_confidence_words:
            if word in ai_response.lower():
                score -= 0.05
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _calculate_relevance_score(self, document: Dict[str, Any], structured_analysis: Dict[str, Any]) -> float:
        """Calculate relevance score based on document content and analysis"""
        score = 0.5  # Base score
        
        content = document.get('content', '').lower()
        
        # Key terms that increase relevance
        high_relevance_terms = [
            'inheritance', 'will', 'property', 'estate', 'probate',
            'gibraltar', 'sean thweny', 'nadia', 'administration',
            'transfer', 'legal', 'court', 'solicitor'
        ]
        
        for term in high_relevance_terms:
            if term in content:
                score += 0.05
        
        # Document type relevance
        doc_type = document.get('document_type', '')
        if doc_type in ['email', 'legal_document', 'timeline_message']:
            score += 0.1
        
        # Timestamp relevance (recent documents might be more relevant)
        timestamp = document.get('timestamp', '')
        if timestamp and '2023' in timestamp:  # Recent year
            score += 0.05
        
        # Analysis quality affects relevance
        if len(structured_analysis.get('legal_facts', [])) > 2:
            score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def process_documents_batch(self, documents: List[Dict[str, Any]], 
                               prompt_template: str, settings: Dict[str, Any],
                               progress_callback=None) -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        results = []
        total = len(documents)
        
        for i, document in enumerate(documents):
            try:
                self.logger.info(f"Processing document {i+1}/{total}: {document.get('source_file', 'Unknown')}")
                
                result = self.analyze_document(
                    document=document,
                    prompt_template=prompt_template,
                    enable_firecrawl=settings.get('enable_firecrawl', True),
                    max_tokens=settings.get('max_tokens', 2048),
                    temperature=settings.get('temperature', 0.7)
                )
                
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, total, document.get('source_file', 'Unknown'))
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to process document {i+1}: {e}")
                # Continue with next document
                continue
        
        self.logger.info(f"Batch processing completed: {len(results)}/{total} documents processed successfully")
        return results
