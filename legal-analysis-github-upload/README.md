# Deep Research Project

A comprehensive legal analysis and AI processing system with advanced MCP (Model Context Protocol) server integration.

## Overview

This project is a sophisticated research and analysis platform that combines:

- **Legal Analysis Engine**: Advanced document processing and legal case analysis
- **AI Integration**: Multiple AI model integrations including Gemini 2.5 Pro
- **Parallel Processing**: High-performance batch processing capabilities
- **MCP Servers**: Comprehensive Model Context Protocol server implementations
- **Real-time Monitoring**: Live analysis and performance monitoring
- **Vector Storage**: Qdrant integration for efficient document retrieval
- **Web Interface**: Streamlit-based user interface

## Key Features

### Legal Analysis
- Document parsing and content extraction
- Case analysis and legal reasoning
- Batch processing of legal documents
- Real-time analysis monitoring

### AI Processing
- Gemini 2.5 Pro integration
- Parallel API processing
- Smart rate limiting and optimization
- Multiple model support

### MCP Integration
- Redis MCP server
- GitLab MCP server
- Memory management systems
- Comprehensive server testing

### Performance & Monitoring
- Real-time performance monitoring
- Database optimization
- Parallel processing optimization
- Advanced debugging tools

## Project Structure

```
├── src/                    # Core source code
│   ├── legal_analysis_engine.py
│   ├── document_parser.py
│   ├── enhanced_message_processor.py
│   └── ...
├── streamlit_app.py        # Main web interface
├── mcp_config.json         # MCP server configuration
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── docs/                   # Documentation files
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jthweny/deep-research.git
   cd deep-research
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## Configuration

### API Keys Required
- **Gemini API Key**: For AI processing
- **Firecrawl API Key**: For web scraping
- **Other service keys**: As specified in `.env`

### MCP Servers
The project includes multiple MCP server configurations:
- Redis server for caching
- GitLab integration
- Memory management
- Custom research tools

## Documentation

- [Implementation Guide](IMPLEMENTATION_COMPLETE.md)
- [MCP Server Status](MCP_STATUS_FINAL.md)
- [Performance Optimization](OPTIMIZATION_SUMMARY.md)
- [System Status](SYSTEM_STATUS_EXPLAINED.md)

## Recent Updates

- ✅ Gemini 2.5 Pro integration complete
- ✅ Parallel processing optimization
- ✅ MCP servers fully configured
- ✅ Real-time monitoring system
- ✅ Database performance improvements

## Technologies Used

- **Python**: Core application language
- **Streamlit**: Web interface
- **Qdrant**: Vector database
- **Redis**: Caching and MCP
- **GitLab API**: Version control integration
- **Google Gemini**: AI processing
- **Docker**: Containerization support

## Contributing

This is a research project focused on advanced legal analysis and AI integration. The codebase demonstrates:

- Advanced parallel processing techniques
- MCP server implementation patterns
- Real-time monitoring systems
- Performance optimization strategies

## License

This project is for research and educational purposes.

## Contact

- Repository: https://github.com/jthweny/deep-research
- Issues: Use GitHub issue tracker for bug reports and feature requests
