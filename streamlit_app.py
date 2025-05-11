import streamlit as st
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re
import time
import nltk
import os

# --- Configuration & Setup ---

# Download nltk sentence tokenizer data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Core Functions ---

def configure_gemini(api_key):
    """Configures the Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        # Test the API key by trying to list models
        try:
            list(genai.list_models()) # Removed models variable as it's not used
            return True
        except Exception as e:
            st.error(f"Invalid API key or API error: {e}")
            return False
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return False

def get_gemini_embeddings(texts, model_name="models/text-embedding-004"):
    """
    Generates embeddings for a list of texts using the Gemini API.

    Args:
        texts (list): A list of strings to embed.
        model_name (str): The name of the embedding model to use.

    Returns:
        numpy.ndarray: An array of embeddings, or an empty array if an error occurs.
    """
    if not texts:
        return np.array([])
    try:
        # Ensure texts are not empty strings, replace if necessary
        processed_texts = [text if text.strip() else "empty" for text in texts]
        
        embeddings_list = []
        for text in processed_texts:
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings_list.append(result['embedding'])
        
        embeddings = np.array(embeddings_list)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

def fetch_blog_content(url):
    """
    Fetches and parses the title and main text content from a blog post URL.

    Args:
        url (str): The URL of the blog post.

    Returns:
        tuple: (title, content_text) or (None, None) if fetching fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(separator=' ', strip=True) if title_tag else "No title found"

        main_content_tags = ['article', 'main', 'main-content', 'post-content', 'entry-content', 'td-post-content']
        content_element = None
        for tag_or_class in main_content_tags:
            content_element = soup.find(class_=tag_or_class) or soup.find(tag_or_class)
            if content_element:
                break
        
        if content_element:
            paragraphs = content_element.find_all('p')
        else:
            paragraphs = soup.find_all('p')
            if not paragraphs:
                 st.warning(f"Could not find specific content tags or paragraphs for {url}. Extracting all text.")
                 content_text = soup.get_text(separator=' ', strip=True)
                 content_text = re.sub(r'\s+', ' ', content_text)
                 return title, content_text

        content_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        content_text = re.sub(r'\s+', ' ', content_text)
        
        if not content_text:
            st.warning(f"Could not extract meaningful paragraph text from {url}. The page structure might be complex.")
            body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
            body_text = re.sub(r'\s+', ' ', body_text)
            return title, body_text

        return title, content_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing {url}: {e}")
        return None, None


def chunk_text_by_sentences(text, min_sentence_length=5):
    """
    Chunks text into sentences using NLTK.

    Args:
        text (str): The text to chunk.
        min_sentence_length (int): Minimum number of words for a sentence to be included.

    Returns:
        list: A list of sentences.
    """
    if not text:
        return []
    sentences = nltk.sent_tokenize(text)
    return [s for s in sentences if len(s.split()) >= min_sentence_length]


def compare_and_get_similar_chunks(source_chunks, target_title_embedding, threshold=0.75):
    """
    Compares source chunk embeddings with the target title embedding.

    Args:
        source_chunks (list): List of text chunks from the source blog.
        target_title_embedding (numpy.ndarray): Embedding of the target blog title.
        threshold (float): Similarity threshold.

    Returns:
        dict: A dictionary of {chunk: similarity_score} for chunks above the threshold.
    """
    if not source_chunks or target_title_embedding is None or target_title_embedding.size == 0:
        return {}
    
    with st.spinner("Generating embeddings for source chunks..."):
        source_chunk_embeddings = get_gemini_embeddings(source_chunks)

    if source_chunk_embeddings is None or source_chunk_embeddings.size == 0:
        st.warning("Could not generate embeddings for source chunks.")
        return {}

    results = {}
    try:
        if target_title_embedding.ndim == 1:
            target_title_embedding = target_title_embedding.reshape(1, -1)

        similarities = cosine_similarity(source_chunk_embeddings, target_title_embedding)
        
        for idx, similarity_score in enumerate(similarities):
            if similarity_score[0] >= threshold:
                results[source_chunks[idx]] = float(similarity_score[0])
        
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        return sorted_results

    except Exception as e:
        st.error(f"Error during similarity comparison: {e}")
        return {}

def get_keyword_suggestions_from_gemini(target_blog_title, similar_chunks_dict, model_name="gemini-1.5-flash"):
    """
    Uses Gemini to suggest keywords from chunks to link to the target blog title.

    Args:
        target_blog_title (str): The title of the target blog post.
        similar_chunks_dict (dict): Dictionary of {chunk: similarity_score}.
        model_name (str): The Gemini model for generation.

    Returns:
        dict: A dictionary of {chunk: suggested_keyword}.
    """
    if not similar_chunks_dict:
        return {}

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error initializing Gemini generative model: {e}")
        return {}

    results = {}
    
    progress_bar = st.progress(0)
    total_chunks = len(similar_chunks_dict)

    for i, (chunk, _) in enumerate(similar_chunks_dict.items()):
        prompt = f"""
        You are an expert in SEO and blog interlinking.
        I have a chunk of text from a blog post that is semantically similar to the title of a target blog post.
        Your task is to identify the best keyword or short phrase (2-5 words) from WITHIN THE PROVIDED CHUNK that can be used as anchor text to link to the target blog post.

        Follow these rules strictly:
        1. The keyword/phrase MUST exist verbatim within the 'Chunk of Text'. Do not invent or rephrase.
        2. The keyword/phrase should be highly relevant to the 'Title of Target Blog Post'.
        3. Choose the most concise and natural-sounding keyword/phrase.
        4. If no suitable keyword/phrase is found within the chunk, respond with "None".
        5. Your response should ONLY be the keyword/phrase itself, or "None". Do not add any other explanations or text.

        Chunk of Text:
        ```{chunk}```

        Title of Target Blog Post:
        ```{target_blog_title}```

        Suggested Keyword/Phrase from Chunk:
        """
        
        try:
            time.sleep(1) 
            
            response = model.generate_content(prompt)
            suggested_keyword = response.text.strip()
            
            if suggested_keyword.lower() not in chunk.lower() and suggested_keyword != "None":
                results[chunk] = f"{suggested_keyword} (Warning: Keyword not found verbatim in chunk by simple check)"
            else:
                 results[chunk] = suggested_keyword

        except Exception as e:
            st.warning(f"Error generating keyword for a chunk: {e}. Skipping this chunk.")
            results[chunk] = "Error generating suggestion"
        
        progress_bar.progress((i + 1) / total_chunks)

    progress_bar.empty()
    return results

# --- Streamlit UI ---

st.set_page_config(page_title="Semantic Interlinking Tool", layout="wide")

st.title("🔗 Semantic Blog Post Interlinking Tool (with Gemini AI)")
st.markdown("""
This tool helps you find interlinking opportunities between your blog posts.
Enter the URL of your **source blog post** (where you want to add links)
and the URL of your **target blog post** (the post you want to link to).
The tool will analyze the source post for relevant text chunks and suggest keywords to link to the target post.
""")

# --- API Key Input ---
st.sidebar.header("🔑 API Configuration")
gemini_api_key = st.sidebar.text_input("Enter your Google AI Studio API Key:", type="password")

st.sidebar.markdown("""
    **How to get your API Key:**
    1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    2. Create a new API key (or use an existing one).
    3. Copy and paste it here.
    
    *Your API key is used to make calls to the Gemini models for embeddings and content generation.
    It is not stored by this application after you close the browser tab (if running locally).*
""")


# --- Main App Inputs ---
st.header("✍️ Input Blog Post URLs")
col1, col2 = st.columns(2)
with col1:
    source_blog_url = st.text_input("Source Blog Post URL (where to add links):", 
                                    placeholder="e.g., https://yourblog.com/source-article")
with col2:
    target_blog_url = st.text_input("Target Blog Post URL (to link to):",
                                    placeholder="e.g., https://yourblog.com/target-article")

similarity_threshold = st.slider("Similarity Threshold:", min_value=0.1, max_value=1.0, value=0.70, step=0.01,
                                 help="Chunks from the source post with similarity to the target title above this value will be considered. Higher means more similar.")


if st.button("🔗 Analyze for Interlinking Opportunities", type="primary", use_container_width=True):
    if not gemini_api_key:
        st.error("🚨 Please enter your Gemini API Key in the sidebar to proceed.")
    elif not source_blog_url or not target_blog_url:
        st.warning("⚠️ Please enter both source and target blog post URLs.")
    elif not (source_blog_url.startswith("http://") or source_blog_url.startswith("https://")) or \
         not (target_blog_url.startswith("http://") or target_blog_url.startswith("https://")):
        st.warning("⚠️ Please enter valid URLs (starting with http:// or https://).")
    else:
        if not configure_gemini(gemini_api_key):
            st.stop() 

        with st.spinner("🚀 Starting analysis... This may take a few moments."):
            
            st.subheader("🎯 Target Blog Post")
            with st.spinner(f"Fetching content from target URL: {target_blog_url}..."):
                target_title, target_content = fetch_blog_content(target_blog_url)
            
            if not target_title or not target_content:
                st.error(f"Could not fetch content for the target URL: {target_blog_url}. Please check the URL and try again.")
                st.stop()
            
            st.markdown(f"**Title:** {target_title}")
            with st.expander("View Target Content (First 500 chars)"):
                st.text(target_content[:500] + "...")

            with st.spinner("Generating embedding for target title..."):
                target_title_embedding = get_gemini_embeddings([target_title])

            if target_title_embedding is None or target_title_embedding.size == 0:
                st.error("Failed to generate embedding for the target blog title.")
                st.stop()

            st.subheader("📄 Source Blog Post")
            with st.spinner(f"Fetching content from source URL: {source_blog_url}..."):
                source_title, source_content = fetch_blog_content(source_blog_url)

            if not source_title or not source_content:
                st.error(f"Could not fetch content for the source URL: {source_blog_url}. Please check the URL and try again.")
                st.stop()
            
            st.markdown(f"**Title:** {source_title}")
            with st.expander("View Source Content (First 500 chars)"):
                st.text(source_content[:500] + "...")

            with st.spinner("Chunking source content into sentences..."):
                source_chunks = chunk_text_by_sentences(source_content)
            
            if not source_chunks:
                st.warning("No suitable text chunks found in the source blog post after filtering.")
                st.stop()
            st.info(f"Extracted {len(source_chunks)} sentences from the source blog post.")

            st.subheader("📊 Similarity Analysis")
            with st.spinner(f"Comparing source chunks with target title (threshold: {similarity_threshold})..."):
                similar_chunks_dict = compare_and_get_similar_chunks(source_chunks, target_title_embedding, threshold=similarity_threshold)

            if not similar_chunks_dict:
                st.warning(f"No chunks found in the source post with similarity above {similarity_threshold} to the target title.")
                st.stop()
            
            st.success(f"Found {len(similar_chunks_dict)} potentially relevant chunks in the source post!")

            st.subheader("💡 Keyword Suggestions for Interlinking")
            with st.spinner("Asking Gemini AI for keyword suggestions (this might take a while for many chunks)..."):
                keyword_suggestions = get_keyword_suggestions_from_gemini(target_title, similar_chunks_dict)

            if not keyword_suggestions:
                st.warning("Could not generate keyword suggestions.")
            else:
                st.success("Generated keyword suggestions!")
                
                results_data = []
                for chunk, keyword in keyword_suggestions.items():
                    similarity = similar_chunks_dict.get(chunk, "N/A")
                    similarity_str = f"{similarity:.2%}" if isinstance(similarity, float) else "N/A"
                    results_data.append({
                        "Source Chunk": chunk,
                        "Suggested Keyword": keyword,
                        "Similarity to Target Title": similarity_str,
                        "Link to": target_blog_url
                    })
                
                st.dataframe(results_data, use_container_width=True)

                st.markdown("---")
                st.balloons()
                st.markdown(f"""
                ### Next Steps:
                1.  **Review the suggestions:** Check if the keywords make sense in context.
                2.  **Verify keywords:** Ensure the "Suggested Keyword" is indeed present in the "Source Chunk". The AI tries its best, but verification is key.
                3.  **Implement links:** Go to your source blog post and add hyperlinks using the suggested keywords, pointing to `{target_blog_url}`.
                """)

st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini. For educational and demonstration purposes.")
st.markdown("Considerations: Web page structures vary. Content extraction might not be perfect for all sites.")
