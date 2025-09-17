# AI-Powered Image Similarity Search

An AI-driven image similarity and search pipeline that combines local dataset search (Azure Cognitive Search) and web-based search (Google Lens via SerpAPI). This tool allows you to input an image (local file or URL) and returns top-K visually similar images from both your indexed dataset and the web.

## Features

🔍 **Dual Search Capability**
- Local dataset search using Azure Cognitive Search
- Web-based search using Google Lens via SerpAPI
- Combined results from both sources

🖼️ **Flexible Image Input**
- Upload local image files
- Provide image URLs
- Support for various image formats (JPEG, PNG, etc.)

⚡ **AI-Powered Similarity**
- Advanced computer vision models for feature extraction
- Cosine similarity calculations for accurate matching
- Top-K results ranking

🔧 **Enterprise-Ready**
- Azure Cognitive Search integration for scalable indexing
- RESTful API endpoints
- Configurable similarity thresholds

## Architecture

The system consists of several key components:

1. **Image Processing Pipeline**: Extracts visual features from input images
2. **Local Search Engine**: Queries indexed dataset using Azure Cognitive Search
3. **Web Search Integration**: Leverages Google Lens through SerpAPI
4. **Similarity Calculator**: Computes similarity scores and ranks results
5. **API Layer**: Provides RESTful endpoints for integration



## Performance Optimization

- **Batch Processing**: Index images in batches for better performance
- **Caching**: Implement Redis caching for frequently accessed results
- **Async Processing**: Use async operations for concurrent searches
- **Feature Precomputation**: Pre-compute and store image features

- 
## Prerequisites

- Python 3.8+
- Azure Cognitive Search service
- SerpAPI account for Google Lens integration
- Required Python packages (see requirements.txt)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kavyanegi0007/AI-Powered-Image-Similarity-Search.git
   cd AI-Powered-Image-Similarity-Search
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Configure the following variables in your `.env` file:
   ```env
   # Azure Cognitive Search
   AZURE_SEARCH_SERVICE_NAME=your-search-service
   AZURE_SEARCH_ADMIN_KEY=your-admin-key
   AZURE_SEARCH_INDEX_NAME=your-index-name
   
   # SerpAPI Configuration
   SERPAPI_KEY=your-serpapi-key
   
   # Application Settings
   MAX_RESULTS=10
   SIMILARITY_THRESHOLD=0.7
   ```

## Configuration

### Azure Cognitive Search Setup

1. Create an Azure Cognitive Search service
2. Create a search index for your image dataset
3. Configure the index schema to include:
   - Image metadata (filename, path, etc.)
   - Feature vectors for similarity search
   - Searchable text descriptions

### SerpAPI Setup

1. Sign up for a SerpAPI account at [serpapi.com](https://serpapi.com)
2. Get your API key from the dashboard
3. Add the key to your environment configuration

## Usage

### Basic Usage

```python
from image_similarity import ImageSimilaritySearch

# Initialize the search engine
search_engine = ImageSimilaritySearch()

# Search with local file
results = search_engine.search_similar('path/to/your/image.jpg', top_k=5)

# Search with URL
results = search_engine.search_similar('https://example.com/image.jpg', top_k=5)

# Print results
for result in results:
    print(f"Source: {result['source']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"URL: {result['url']}")
    print("---")
```

### API Usage

Start the API server:
```bash
python app.py
```

Search for similar images using POST request:
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "top_k": 10,
    "include_web": true,
    "include_local": true
  }'
```

### Web Interface

Access the web interface at `http://localhost:5000` to:
- Upload images through a user-friendly interface
- View search results with thumbnails
- Adjust search parameters
- Export results

## API Endpoints

### POST /api/search
Search for similar images

**Request Body:**
```json
{
  "image_path": "string (optional)",
  "image_url": "string (optional)",
  "image_data": "string (base64, optional)",
  "top_k": "integer (default: 10)",
  "include_local": "boolean (default: true)",
  "include_web": "boolean (default: true)",
  "similarity_threshold": "float (default: 0.7)"
}
```

**Response:**
```json
{
  "query_image": "string",
  "total_results": "integer",
  "local_results": [
    {
      "id": "string",
      "similarity": "float",
      "url": "string",
      "metadata": "object"
    }
  ],
  "web_results": [
    {
      "title": "string",
      "url": "string",
      "source": "string",
      "similarity": "float"
    }
  ]
}
```


## Troubleshooting

### Common Issues

1. **Azure Search Connection Failed**
   - Verify your service name and admin key
   - Check network connectivity and firewall rules

2. **SerpAPI Rate Limits**
   - Monitor your API usage
   - Implement request throttling
   - Consider upgrading your plan

3. **High Memory Usage**
   - Reduce batch sizes for indexing
   - Use smaller image resolutions
   - Implement memory monitoring

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Azure Cognitive Search for scalable indexing
- SerpAPI for web search capabilities
- OpenAI CLIP for advanced image understanding
- The open-source computer vision community

## Contact

- **Author**: Kavya Negi
- **GitHub**: [@Kavyanegi0007](https://github.com/Kavyanegi0007)
- **Project Link**: [https://github.com/Kavyanegi0007/AI-Powered-Image-Similarity-Search](https://github.com/Kavyanegi0007/AI-Powered-Image-Similarity-Search)

## Roadmap

- [ ] Support for video similarity search
- [ ] Real-time similarity search
- [ ] Mobile app integration
- [ ] Advanced filtering options
- [ ] Multi-modal search (text + image)
- [ ] Performance analytics dashboard

---

⭐ If you found this project helpful, please give it a star!
