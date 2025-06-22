"""
Legal researcher that combines message analysis with Firecrawl research.
Orchestrates the research process for messages needing legal information.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .hybrid_database import HybridDatabaseManager
from .smart_api_manager import SmartAPIManager
from .firecrawl_client import FirecrawlClient

console = Console()


class LegalResearcher:
    """Orchestrates legal research for messages that need it."""
    
    def __init__(self, db_manager: HybridDatabaseManager,
                 api_manager: SmartAPIManager,
                 firecrawl_client: FirecrawlClient):
        self.db = db_manager
        self.api = api_manager
        self.firecrawl = firecrawl_client
        
        # Research synthesis prompt
        self.synthesis_prompt = """
Based on the following legal research findings, provide a comprehensive analysis 
relevant to this message from the probate case:

Original Message Summary: {message_summary}
Legal Topic: {legal_topic}
Jurisdiction: {jurisdiction}

Research Findings:
{research_content}

Provide your analysis in JSON format:
{{
    "key_legal_principles": ["principle1", "principle2"],
    "applicable_laws": ["law1", "law2"],
    "tax_implications": "description of tax consequences",
    "procedural_requirements": ["requirement1", "requirement2"],
    "deadlines": ["deadline1", "deadline2"],
    "risks": ["risk1", "risk2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "confidence_level": "high/medium/low",
    "further_research_needed": ["topic1", "topic2"]
}}

Focus on practical implications for the case, especially regarding:
- Gibraltar vs UK tax residency
- Property transfer validity
- Administrative delays
- Inheritance tax liability
"""
    
    async def research_pending_messages(self) -> Dict[str, Any]:
        """Research all messages marked as needing legal research."""
        stats = {
            "messages_researched": 0,
            "research_successful": 0,
            "research_failed": 0,
            "synthesis_completed": 0
        }
        
        # Log start
        await self.db.log_processing_phase(
            phase="legal_research",
            status="started",
            start_time=datetime.now()
        )
        
        # Get messages needing research
        messages = await self.db.get_messages_needing_research()
        
        console.print(f"\n[bold blue]Found {len(messages)} messages needing legal research[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Researching legal topics...", total=len(messages))
            
            for msg in messages:
                try:
                    # Extract research parameters
                    research_params = {
                        "query": msg.get("legal_topic", ""),
                        "jurisdiction": msg.get("jurisdiction", "uk").lower(),
                        "context": msg.get("brief_summary", "")
                    }
                    
                    # Skip if no legal topic
                    if not research_params["query"]:
                        progress.update(task, advance=1)
                        continue
                    
                    progress.update(
                        task, 
                        description=f"Researching: {research_params['query'][:50]}..."
                    )
                    
                    # Perform Firecrawl research
                    research_result = await self.firecrawl.search_legal_info(
                        query=research_params["query"],
                        jurisdiction=research_params["jurisdiction"],
                        context=research_params["context"]
                    )
                    
                    if research_result:
                        stats["research_successful"] += 1
                        
                        # Synthesize findings with Gemini
                        synthesis = await self._synthesize_research(
                            msg, research_result
                        )
                        
                        if synthesis:
                            stats["synthesis_completed"] += 1
                            
                            # Save research and synthesis
                            await self._save_research_findings(
                                msg["message_id"],
                                research_params,
                                research_result,
                                synthesis
                            )
                        
                    else:
                        stats["research_failed"] += 1
                        logger.warning(f"No research results for: {research_params['query']}")
                    
                    stats["messages_researched"] += 1
                    progress.update(task, advance=1)
                    
                    # Rate limiting
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Error researching message {msg.get('message_id')}: {e}")
                    stats["research_failed"] += 1
                    progress.update(task, advance=1)
        
        # Log completion
        await self.db.log_processing_phase(
            phase="legal_research",
            status="completed",
            end_time=datetime.now(),
            items_processed=stats["messages_researched"],
            errors=stats["research_failed"],
            metadata=stats
        )
        
        # Display summary
        self._display_research_summary(stats)
        
        return stats
    
    async def _synthesize_research(self, 
                                  message: Dict[str, Any],
                                  research: Dict[str, Any]) -> Optional[Dict]:
        """Synthesize research findings using Gemini."""
        try:
            # Prepare research content
            research_content = self._format_research_content(research)
            
            # Create synthesis prompt
            prompt = self.synthesis_prompt.format(
                message_summary=message.get("brief_summary", ""),
                legal_topic=message.get("legal_topic", ""),
                jurisdiction=message.get("jurisdiction", ""),
                research_content=research_content
            )
            
            # Call Gemini for synthesis
            response = await self.api.generate_content(
                prompt=prompt,
                system_instruction="You are a legal expert synthesizing research findings. "
                                 "Focus on practical implications for the probate case.",
                temperature=0.3,
                max_tokens=1500
            )
            
            if not response:
                return None
            
            # Parse JSON response
            try:
                # Clean response
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                
                synthesis = json.loads(cleaned.strip())
                return synthesis
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse synthesis JSON: {e}")
                return {
                    "error": "Failed to parse synthesis",
                    "raw_response": response[:500]
                }
                
        except Exception as e:
            logger.error(f"Error synthesizing research: {e}")
            return None
    
    def _format_research_content(self, research: Dict[str, Any]) -> str:
        """Format research findings for synthesis."""
        content_parts = []
        
        # Add search results summary
        search_results = research.get("search_results", [])
        if search_results:
            content_parts.append("Search Results:")
            for idx, result in enumerate(search_results[:5], 1):
                content_parts.append(f"{idx}. {result.get('title', 'No title')}")
                content_parts.append(f"   URL: {result.get('url', '')}")
                content_parts.append("")
        
        # Add scraped content
        scraped = research.get("scraped_content", [])
        if scraped:
            content_parts.append("\nDetailed Findings:")
            for idx, item in enumerate(scraped, 1):
                content_parts.append(f"\nSource {idx}: {item.get('title', 'No title')}")
                content_parts.append(f"URL: {item.get('url', '')}")
                content_parts.append("Content Extract:")
                
                # Get first 1000 chars of content
                content = item.get("content", "")[:1000]
                content_parts.append(content)
                
                # Extract legal provisions
                provisions = self.firecrawl.extract_legal_provisions(
                    item.get("content", "")
                )
                
                if any(provisions.values()):
                    content_parts.append("\nExtracted Provisions:")
                    for prov_type, items in provisions.items():
                        if items:
                            content_parts.append(f"- {prov_type}: {', '.join(items[:3])}")
                
                content_parts.append("")
        
        return "\n".join(content_parts)
    
    async def _save_research_findings(self,
                                     message_id: str,
                                     params: Dict,
                                     research: Dict,
                                     synthesis: Dict):
        """Save research findings to database."""
        try:
            # Prepare research data
            research_data = {
                "query": params["query"],
                "jurisdiction": params["jurisdiction"],
                "findings": synthesis,
                "sources": [
                    {
                        "url": item.get("url"),
                        "title": item.get("title")
                    }
                    for item in research.get("scraped_content", [])
                ]
            }
            
            # Save to database
            await self.db.save_legal_research(message_id, research_data)
            
            logger.info(f"Saved research for message {message_id}")
            
        except Exception as e:
            logger.error(f"Error saving research findings: {e}")
    
    def _display_research_summary(self, stats: Dict[str, Any]):
        """Display research summary in a table."""
        table = Table(title="Legal Research Summary")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Messages Researched", str(stats["messages_researched"]))
        table.add_row("Research Successful", str(stats["research_successful"]))
        table.add_row("Research Failed", str(stats["research_failed"]))
        table.add_row("Synthesis Completed", str(stats["synthesis_completed"]))
        
        console.print("\n")
        console.print(table)
    
    async def generate_research_report(self) -> str:
        """Generate a comprehensive research report."""
        # Get all messages with research
        cursor = await self.db._connection.execute("""
            SELECT ma.*, lr.findings, lr.sources
            FROM message_analysis ma
            JOIN legal_research lr ON ma.message_id = lr.message_id
            WHERE ma.is_relevant = 1
            ORDER BY ma.relevance_score DESC
        """)
        
        messages_with_research = await cursor.fetchall()
        
        # Build report
        report_parts = [
            "# Legal Research Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Executive Summary",
            f"Total messages researched: {len(messages_with_research)}",
            "",
            "## Key Legal Findings",
            ""
        ]
        
        # Group by jurisdiction
        by_jurisdiction = {}
        for msg in messages_with_research:
            jurisdiction = msg["jurisdiction"] or "Unknown"
            if jurisdiction not in by_jurisdiction:
                by_jurisdiction[jurisdiction] = []
            by_jurisdiction[jurisdiction].append(dict(msg))
        
        # Add findings by jurisdiction
        for jurisdiction, messages in by_jurisdiction.items():
            report_parts.append(f"### {jurisdiction.upper()} Law")
            report_parts.append("")
            
            for msg in messages[:5]:  # Top 5 per jurisdiction
                report_parts.append(f"**Topic**: {msg.get('legal_topic', 'N/A')}")
                report_parts.append(f"**Message Summary**: {msg.get('brief_summary', 'N/A')}")
                
                # Parse findings
                try:
                    findings = json.loads(msg.get("findings", "{}"))
                    
                    if findings.get("key_legal_principles"):
                        report_parts.append("**Key Principles**:")
                        for principle in findings["key_legal_principles"]:
                            report_parts.append(f"- {principle}")
                    
                    if findings.get("tax_implications"):
                        report_parts.append(f"**Tax Implications**: {findings['tax_implications']}")
                    
                    if findings.get("risks"):
                        report_parts.append("**Risks**:")
                        for risk in findings["risks"]:
                            report_parts.append(f"- {risk}")
                    
                    report_parts.append("")
                    
                except:
                    pass
        
        # Add recommendations section
        report_parts.extend([
            "## Consolidated Recommendations",
            "",
            "Based on the legal research conducted:",
            ""
        ])
        
        # Collect all recommendations
        all_recommendations = set()
        for msg in messages_with_research:
            try:
                findings = json.loads(msg.get("findings", "{}"))
                for rec in findings.get("recommendations", []):
                    all_recommendations.add(rec)
            except:
                pass
        
        for rec in sorted(all_recommendations):
            report_parts.append(f"- {rec}")
        
        report = "\n".join(report_parts)
        
        # Save report
        report_path = "data/legal_research_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        console.print(f"\n[green]Report saved to: {report_path}[/green]")
        
        return report


# Main execution
async def main():
    """Run legal research on pending messages."""
    # Initialize components
    db = DatabaseManager()
    await db.initialize()
    
    api = GeminiAPIManager()
    firecrawl = FirecrawlClient()
    
    # Create researcher
    researcher = LegalResearcher(db, api, firecrawl)
    
    # Research pending messages
    await researcher.research_pending_messages()
    
    # Generate report
    await researcher.generate_research_report()
    
    # Close database
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())