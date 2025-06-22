"""
Legal Document Analysis Engine for Case Analysis System
Provides comprehensive AI-powered analysis of legal documents using Gemini 2.5 Pro
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
from datetime import datetime

@dataclass
class LegalAnalysisResult:
    """Structured result of legal document analysis"""
    # Core analysis
    relevance_score: float  # 0.0 - 1.0
    confidence_score: float  # 0.0 - 1.0
    summary: str
    
    # Legal elements
    key_facts: List[str]
    legal_issues: List[str]
    evidence_value: str  # "high", "medium", "low"
    jurisdiction: Optional[str]
    
    # Timeline and context
    timeline_relevance: bool
    chronological_context: str
    contradictions: List[str]
    
    # Property and inheritance specific
    mentions_property: bool
    mentions_inheritance: bool
    mentions_wills: bool
    mentions_tax: bool
    mentions_domicile: bool
    
    # Research needs
    needs_legal_research: bool
    research_topics: List[str]
    legal_citations_needed: List[str]
    
    # Processing metadata
    analysis_type: str
    processing_time: float
    model_used: str
    tokens_used: Optional[int]

class LegalAnalysisEngine:
    """
    AI-powered legal analysis engine using Gemini 2.5 Pro
    Specialized for inheritance/probate case analysis
    """
    
    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.logger = logging.getLogger(__name__)
        
        # Analysis prompts
        self.analysis_prompts = {
            "comprehensive": self._get_comprehensive_prompt(),
            "fact_extraction": self._get_fact_extraction_prompt(),
            "legal_issues": self._get_legal_issues_prompt(),
            "evidence_evaluation": self._get_evidence_evaluation_prompt(),
            "timeline_analysis": self._get_timeline_analysis_prompt()
        }
        
    def _get_comprehensive_prompt(self) -> str:
        """Get the comprehensive legal analysis prompt"""
        return """You are a senior legal analyst specializing in inheritance and probate law. Analyze this document for the Sean Thweny inheritance case.

**CASE CONTEXT:**
- Deceased: Sean Thweny
- Complex inheritance dispute involving properties in Gibraltar and UK
- Issues around validity of wills, property transfers, and administrative procedures
- Key players: Various family members, solicitors, administrators

**ANALYSIS REQUIREMENTS:**
Provide a structured JSON response with the following fields:

1. **relevance_score** (0.0-1.0): How relevant is this document to the inheritance case?
2. **confidence_score** (0.0-1.0): How confident are you in this analysis?
3. **summary** (string): 2-3 sentence summary of document significance
4. **key_facts** (array): Extract 3-5 key factual points
5. **legal_issues** (array): Identify legal matters (inheritance rights, property disputes, procedural issues)
6. **evidence_value** (string): "high", "medium", or "low" - significance as evidence
7. **jurisdiction** (string or null): Legal jurisdiction mentioned (UK, Gibraltar, etc.)
8. **timeline_relevance** (boolean): Does this fit into chronological sequence?
9. **chronological_context** (string): When/how this fits in timeline
10. **contradictions** (array): Any inconsistencies with typical inheritance procedures
11. **mentions_property** (boolean): References to property/real estate
12. **mentions_inheritance** (boolean): Direct inheritance references
13. **mentions_wills** (boolean): Will/testament references
14. **mentions_tax** (boolean): Tax implications mentioned
15. **mentions_domicile** (boolean): Domicile/residence issues
16. **needs_legal_research** (boolean): Requires additional legal research
17. **research_topics** (array): Topics needing further research
18. **legal_citations_needed** (array): Areas needing case law/statute research

**DOCUMENT TO ANALYZE:**
{document_content}

**BASELINE REFERENCE (if available):**
{baseline_content}

Respond ONLY with valid JSON - no other text."""

    def _get_fact_extraction_prompt(self) -> str:
        """Get the fact extraction focused prompt"""
        return """Extract key facts from this legal document for the Sean Thweny inheritance case.

Focus on:
- Names, dates, locations
- Property descriptions and values
- Financial amounts
- Legal procedures and deadlines
- Communication chains

Document: {document_content}

Respond with JSON containing 'facts' array."""

    def _get_legal_issues_prompt(self) -> str:
        """Get the legal issues identification prompt"""
        return """Identify legal issues in this document for the Sean Thweny inheritance case.

Consider:
- Inheritance law violations
- Procedural irregularities  
- Jurisdictional conflicts
- Property law issues
- Tax implications

Document: {document_content}

Respond with JSON containing 'legal_issues' array."""

    def _get_evidence_evaluation_prompt(self) -> str:
        """Get the evidence evaluation prompt"""
        return """Evaluate this document as evidence in the Sean Thweny inheritance case.

Assess:
- Reliability and authenticity
- Relevance to inheritance claims
- Strength for legal proceedings
- Potential weaknesses

Document: {document_content}

Respond with JSON containing evidence evaluation."""

    def _get_timeline_analysis_prompt(self) -> str:
        """Get the timeline analysis prompt"""
        return """Analyze this document's timeline significance for the Sean Thweny inheritance case.

Consider:
- When events occurred
- Sequence relative to other events
- Deadline implications
- Chronological inconsistencies

Document: {document_content}

Respond with JSON containing timeline analysis."""

    def analyze_document(self, document_content: str, analysis_type: str = "comprehensive",
                             baseline_content: Optional[str] = None) -> LegalAnalysisResult:
        """
        Perform comprehensive legal analysis of a document
        
        Args:
            document_content: The document text to analyze
            analysis_type: Type of analysis to perform
            baseline_content: Reference document content for context
            
        Returns:
            LegalAnalysisResult with structured analysis
        """
        start_time = time.time()
        
        # Get the appropriate prompt
        prompt_template = self.analysis_prompts.get(analysis_type, self.analysis_prompts["comprehensive"])
        
        # Format the prompt
        prompt = prompt_template.format(
            document_content=document_content[:8000],  # Limit content length
            baseline_content=baseline_content[:2000] if baseline_content else "Not available"
        )
        
        try:
            # Get analysis from Gemini
            response_text = self._call_gemini(prompt)
            
            # Parse JSON response
            analysis_data = self._parse_analysis_response(response_text)
            
            # Create structured result
            result = LegalAnalysisResult(
                # Core analysis
                relevance_score=analysis_data.get('relevance_score', 0.0),
                confidence_score=analysis_data.get('confidence_score', 0.0),
                summary=analysis_data.get('summary', 'No summary provided'),
                
                # Legal elements
                key_facts=analysis_data.get('key_facts', []),
                legal_issues=analysis_data.get('legal_issues', []),
                evidence_value=analysis_data.get('evidence_value', 'medium'),
                jurisdiction=analysis_data.get('jurisdiction'),
                
                # Timeline and context
                timeline_relevance=analysis_data.get('timeline_relevance', False),
                chronological_context=analysis_data.get('chronological_context', ''),
                contradictions=analysis_data.get('contradictions', []),
                
                # Property and inheritance specific
                mentions_property=analysis_data.get('mentions_property', False),
                mentions_inheritance=analysis_data.get('mentions_inheritance', False),
                mentions_wills=analysis_data.get('mentions_wills', False),
                mentions_tax=analysis_data.get('mentions_tax', False),
                mentions_domicile=analysis_data.get('mentions_domicile', False),
                
                # Research needs
                needs_legal_research=analysis_data.get('needs_legal_research', False),
                research_topics=analysis_data.get('research_topics', []),
                legal_citations_needed=analysis_data.get('legal_citations_needed', []),
                
                # Processing metadata
                analysis_type=analysis_type,
                processing_time=time.time() - start_time,
                model_used=self.api_manager.get_current_model(),
                tokens_used=None  # Would need to be extracted from API response
            )
            
            self.logger.info(f"✅ Analysis completed: {result.relevance_score:.2f} relevance, {result.confidence_score:.2f} confidence")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            # Return a fallback analysis
            return self._create_fallback_result(analysis_type, time.time() - start_time, str(e))

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with smart retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Use synchronous API manager method (not async)
                response_text, metadata = self.api_manager.generate_content_sync(
                    prompt,
                    max_output_tokens=4096,
                    temperature=0.1  # Low temperature for consistent legal analysis
                )
                
                self.logger.info(f"Generated response using {metadata.get('key_type')} key in {metadata.get('processing_time', 0):.2f}s")
                
                if response_text and response_text.strip():
                    return response_text
                else:
                    raise Exception("Empty response from Gemini")
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"All Gemini API attempts failed: {e}")
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        raise Exception("Failed to get response from Gemini API")

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate the JSON response from Gemini"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                return json.loads(json_text)
            else:
                # Fallback: try parsing entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response text: {response_text[:500]}...")
            
            # Return a fallback structure
            return {
                'relevance_score': 0.5,
                'confidence_score': 0.3,
                'summary': 'Analysis failed to parse - manual review needed',
                'key_facts': ['Analysis parsing failed'],
                'legal_issues': ['Unable to determine'],
                'evidence_value': 'unknown',
                'needs_legal_research': True,
                'research_topics': ['Manual analysis required']
            }

    def _create_fallback_result(self, analysis_type: str, processing_time: float, error_msg: str) -> LegalAnalysisResult:
        """Create a fallback result when analysis fails"""
        return LegalAnalysisResult(
            relevance_score=0.0,
            confidence_score=0.0,
            summary=f"Analysis failed: {error_msg}",
            key_facts=[],
            legal_issues=[],
            evidence_value="unknown",
            jurisdiction=None,
            timeline_relevance=False,
            chronological_context="",
            contradictions=[],
            mentions_property=False,
            mentions_inheritance=False,
            mentions_wills=False,
            mentions_tax=False,
            mentions_domicile=False,
            needs_legal_research=True,
            research_topics=["Manual analysis required"],
            legal_citations_needed=[],
            analysis_type=analysis_type,
            processing_time=processing_time,
            model_used="failed",
            tokens_used=None
        )

    def analyze_batch(self, documents: List[Tuple[str, str]], analysis_type: str = "comprehensive",
                     baseline_content: Optional[str] = None) -> List[LegalAnalysisResult]:
        """
        Analyze multiple documents in batch
        
        Args:
            documents: List of (doc_id, content) tuples
            analysis_type: Type of analysis to perform
            baseline_content: Reference document content
            
        Returns:
            List of LegalAnalysisResult objects
        """
        results = []
        
        for doc_id, content in documents:
            self.logger.info(f"🔍 Analyzing document {doc_id}...")
            
            try:
                # Call synchronous analyze_document method
                result = self.analyze_document(content, analysis_type, baseline_content)
                results.append((doc_id, result))
                
            except Exception as e:
                self.logger.error(f"❌ Failed to analyze document {doc_id}: {e}")
                fallback = self._create_fallback_result(analysis_type, 0.0, str(e))
                results.append((doc_id, fallback))
        
        return results
