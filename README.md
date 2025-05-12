# Blog Interlinker: AI-Powered Content Generator & Internal Linking Tool

This Streamlit application leverages Google's Gemini AI to automatically generate SEO-optimized blog posts and create intelligent internal linking structures. It analyzes your existing website content to maintain consistent brand voice and topic relevance.

## Features

- 🔍 **Website Analysis**: Automatically extracts and analyzes your website's content to understand:
  - Main topic/industry focus
  - Existing brand voice
  - Key themes and keywords
  - Content structure patterns

- ✍️ **Content Generation**:
  - Generates multiple blog post topics aligned with your website's theme
  - Creates full blog post content with proper SEO structure
  - Maintains consistent brand voice and style
  - Supports custom keyword integration

- 🔗 **Smart Internal Linking**:
  - Automatically identifies linking opportunities between posts
  - Creates natural anchor text suggestions
  - Generates a visual linking map
  - Optimizes internal link distribution

- 📦 **WordPress-Ready Export**:
  - Exports content in WordPress-compatible HTML format
  - Includes metadata CSV for easy importing
  - Provides detailed import instructions
  - Maintains all internal linking structures

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blog_interlinker_2.git
cd blog_interlinker_2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get your Google AI Studio API key:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Keep it handy for running the application

## Usage

1. Run the application:
```bash
streamlit run streamlit_app.py
```

2. Enter your Google AI Studio API key in the sidebar

3. Input your website URL and configure generation parameters:
   - Number of posts to generate
   - Preferred brand voice
   - Word count requirements
   - Internal linking density

4. Optional: Upload a CSV file with target keywords

5. Click "Generate Blog Posts" and wait for the magic to happen!

## Advanced Configuration

The application supports several advanced settings:

- **Word Count**: Customize minimum and maximum word counts per post
- **Internal Link Density**: Adjust the target percentage of content that should be internal links
- **Brand Voice**: Choose from multiple voice options (formal, casual, technical, friendly, expert)
- **Custom Keywords**: Upload a CSV file with preferred keywords to target

## Output Format

The application generates a ZIP file containing:

1. WordPress-ready HTML files for each post
2. metadata.csv with post details
3. Internal linking map visualization
4. Import instructions

## Development

This project uses:
- Streamlit for the web interface
- Google Gemini AI for content generation and analysis
- NetworkX for link mapping
- BeautifulSoup for web scraping
- NLTK for text processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit and Google Gemini AI
- Uses various open-source libraries for text processing and analysis
- Inspired by the need for better content generation and internal linking tools

## Live Demo

Try it out at: [Streamlit Cloud URL]

## Support

For issues and feature requests, please use the GitHub issue tracker.
