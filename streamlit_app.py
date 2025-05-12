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
import pandas as pd
import networkx as nx
import markdown
import zipfile # Added zipfile
import csv # Added csv
import io # For BytesIO

# --- Configuration & Setup ---

# Download nltk sentence tokenizer data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

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

def analyze_website_with_gemini(text_content, brand_voice_preference, model_name="gemini-1.5-flash"):
    """
    Analyzes website content using Gemini to extract themes, keywords, and assess brand voice.

    Args:
        text_content (str): The text content scraped from the website.
        brand_voice_preference (str): The user's preferred brand voice (e.g., formal, casual).
        model_name (str): The Gemini model for generation.

    Returns:
        dict: A dictionary containing 'main_topic', 'keywords' (list), 
              'extracted_brand_voice', and 'voice_alignment_notes'.
              Returns None if an error occurs.
    """
    if not text_content:
        st.warning("No text content provided for analysis.")
        return None

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error initializing Gemini generative model for website analysis: {e}")
        return None

    prompt = f"""
    You are an expert SEO and content strategist. Analyze the following website content.
    My goal is to generate blog posts that align with this website's topic, industry, and brand voice.
    The user has indicated a preferred brand voice of '{brand_voice_preference}'.

    Based on the provided text:
    1.  **Main Topic/Industry:** Identify the primary subject matter or industry of the website.
    2.  **Key Themes/Keywords:** List up to 10-15 key themes or keywords that are prominent.
    3.  **Extracted Brand Voice:** Describe the brand voice you observe in the text (e.g., professional, friendly, technical, humorous, sales-oriented).
    4.  **Voice Alignment Notes:** Briefly comment on how the observed voice aligns with the user's preferred voice of '{brand_voice_preference}' and any suggestions for generating new content.

    Website Content:
    ```
    {text_content[:15000]} 
    ```
    (Content truncated to the first 15000 characters for brevity if longer)

    Provide your analysis in a structured format. For example:
    Main Topic: [Your identified topic]
    Keywords: [keyword1, keyword2, keyword3, ...]
    Extracted Brand Voice: [Your description of the voice]
    Voice Alignment Notes: [Your notes on alignment and suggestions]
    
    Focus on extracting actionable insights for content generation.
    """
    # Limit content length for the prompt to avoid exceeding token limits easily
    # The 15000 char limit is an example; actual limits depend on the model.
    
    try:
        st.info("Asking Gemini to analyze website content... this may take a moment.")
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()

        # Parse the structured response
        analysis_data = {}
        for line in analysis_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(" ", "_").replace("/", "_")
                value = value.strip()
                if key == "keywords":
                    analysis_data[key] = [k.strip() for k in value.split(',')]
                else:
                    analysis_data[key] = value
        
        if not all(k in analysis_data for k in ["main_topic", "keywords", "extracted_brand_voice", "voice_alignment_notes"]):
            st.warning("Could not fully parse Gemini's analysis. Using raw response.")
            analysis_data['raw_response'] = analysis_text # Store raw if parsing fails

        return analysis_data

    except Exception as e:
        st.error(f"Error during Gemini website analysis: {e}")
        return None

def generate_blog_topics_with_gemini(analyzed_data, num_topics, brand_voice, user_keywords_df=None, model_name="gemini-1.5-flash"):
    """
    Generates blog post topics using Gemini based on website analysis and user keywords.

    Args:
        analyzed_data (dict): Output from analyze_website_with_gemini.
        num_topics (int): Number of topics to generate.
        brand_voice (str): The desired brand voice.
        user_keywords_df (pd.DataFrame, optional): DataFrame of user-provided keywords.
        model_name (str): The Gemini model for generation.

    Returns:
        list: A list of blog topic strings, or None if an error occurs.
    """
    if not analyzed_data:
        st.warning("No analyzed data provided for topic generation.")
        return None

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error initializing Gemini generative model for topic generation: {e}")
        return None

    main_topic = analyzed_data.get("main_topic", "the website's subject")
    site_keywords = analyzed_data.get("keywords", [])
    
    keyword_prompt_segment = "Focus on these primary keywords from the website analysis: " + ", ".join(site_keywords) + "." if site_keywords else "The website analysis did not yield specific keywords, so focus on the main topic."

    if user_keywords_df is not None and not user_keywords_df.empty:
        user_kw_list = []
        for col in user_keywords_df.columns: # Assuming keywords can be in any column
            user_kw_list.extend(user_keywords_df[col].dropna().astype(str).tolist())
        if user_kw_list:
            keyword_prompt_segment += "\nAdditionally, consider these user-provided keywords: " + ", ".join(list(set(user_kw_list))) + "."


    prompt = f"""
    You are an expert SEO content strategist and blog idea generator.
    I need to generate {num_topics} unique and engaging blog post topics that are HIGHLY RELEVANT to this specific website.

    Website Analysis:
    - Main Topic/Industry: '{main_topic}'
    - Brand Voice: '{brand_voice}'
    - Observed Brand Voice: '{analyzed_data.get("extracted_brand_voice", "Not specified")}'
    - Voice Alignment Notes: '{analyzed_data.get("voice_alignment_notes", "Not specified")}'

    {keyword_prompt_segment}

    Please generate {num_topics} blog post titles that:
    1. Are DIRECTLY related to the website's main topic and core themes
    2. Naturally incorporate the identified keywords where relevant
    3. Match the website's existing content style and depth
    4. Follow SEO best practices (clear, concise titles)
    5. Maintain consistent brand voice
    6. Cover different aspects/angles of the main topic
    7. Build upon the website's existing content themes

    Each title should feel like it belongs on this specific website, not just general topics in the industry.

    List each topic on a new line, without numbering or any other formatting.
    For example:
    The Ultimate Guide to X
    Why Y is Crucial for Z
    10 Tips for Improving A
    """
    
    try:
        st.info(f"Asking Gemini to generate {num_topics} blog topics... this may take a moment.")
        response = model.generate_content(prompt)
        topics_text = response.text.strip()
        generated_topics = [topic.strip() for topic in topics_text.split('\n') if topic.strip()]

        if not generated_topics:
            st.warning("Gemini did not return any topics. The response might have been empty or in an unexpected format.")
            return []
        if len(generated_topics) < num_topics:
            st.warning(f"Gemini returned {len(generated_topics)} topics, which is less than the requested {num_topics}. Using what was returned.")
        
        return generated_topics[:num_topics] # Return up to num_topics

    except Exception as e:
        st.error(f"Error during Gemini topic generation: {e}")
        return None

def generate_post_content_with_gemini(topic_title, analyzed_data, brand_voice, min_words, max_words, existing_keywords=None, model_name="gemini-1.5-flash"):
    """
    Generates blog post content for a given topic using Gemini.

    Args:
        topic_title (str): The title of the blog post.
        analyzed_data (dict): Data from website analysis.
        brand_voice (str): Desired brand voice.
        min_words (int): Minimum word count for the post.
        max_words (int): Maximum word count for the post.
        existing_keywords (list, optional): Keywords from site analysis or user input.
        model_name (str): The Gemini model for generation.

    Returns:
        tuple: (generated_content_str, meta_description_str) or (None, None) if error.
    """
    if not topic_title:
        st.warning("No topic title provided for content generation.")
        return None, None

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error initializing Gemini generative model for content generation: {e}")
        return None, None

    main_site_topic = analyzed_data.get("main_topic", "the website's core subject")
    site_keywords_list = analyzed_data.get("keywords", [])
    if existing_keywords: # Combine site keywords with any other provided keywords
        site_keywords_list = list(set(site_keywords_list + existing_keywords))
    
    keyword_instructions = ""
    if site_keywords_list:
        keyword_instructions = f"Subtly incorporate some of these keywords/themes where natural: {', '.join(site_keywords_list[:5])} (and others from the broader list if relevant)."


    prompt = f"""
    You are an expert blog post writer, skilled in SEO and crafting engaging content.
    Your task is to write a comprehensive blog post that aligns perfectly with the website's existing content and style.

    Blog Post Title: "{topic_title}"

    Website Context:
    - Main Topic/Industry: {main_site_topic}
    - Brand Voice: {brand_voice}
    - Observed Website Voice: {analyzed_data.get("extracted_brand_voice", "Not specified")}
    - Voice Alignment Notes: {analyzed_data.get("voice_alignment_notes", "Not specified")}
    - Target Word Count: Between {min_words} and {max_words} words

    Keywords and Theme Guidance:
    {keyword_instructions}
    Main Website Keywords: {', '.join(analyzed_data.get('keywords', [])[:10])}

    Content Requirements:
    1.  **Introduction:**
        *   Start with a compelling introduction that aligns with the website's style
        *   Clearly establish the connection to the website's main topic/industry
        *   Naturally incorporate primary keywords from the website analysis
        *   Set the tone matching the observed brand voice

    2.  **Main Body:**
        *   Structure content similar to the website's existing format
        *   Use clear headings (H2, H3) that reflect the website's content hierarchy
        *   Maintain consistent depth and detail level with existing content
        *   Provide valuable information that builds upon the website's expertise
        *   Ensure seamless integration with existing content themes
        *   Keep the established '{brand_voice}' tone throughout

    3.  **SEO and Keyword Integration:**
        *   Naturally weave in keywords from both the title and website analysis
        *   Use similar keyword density as observed in the website
        *   Maintain the website's approach to heading optimization
        *   Create opportunities for internal linking with related topics

    4.  **Conclusion:**
        *   Wrap up in a style consistent with the website's other posts
        *   Reinforce the connection to the main topic
        *   Include calls-to-action that match the website's engagement style

    5.  **Meta Description:**
        *   Create a compelling 150-160 character description
        *   Match the website's meta description style
        *   Include primary keywords naturally
        *   Maintain brand voice consistency

    Formatting:
    - Use Markdown for the entire blog post body (including headings).
    - After the full blog post content, on a new line, write "META_DESCRIPTION:" followed by the meta description text.

    Example of H2/H3 usage in Markdown:
    ## This is a Main Section (H2)
    Some text here...
    ### This is a Sub-Section (H3)
    More text here...

    Begin writing the blog post now:
    """

    try:
        # Add a small delay to respect potential API rate limits if calling in a loop
        time.sleep(1.5) # Increased delay slightly for content generation
        
        st.info(f"Asking Gemini to generate content for '{topic_title}'...")
        response = model.generate_content(prompt)
        full_response_text = response.text.strip()

        # Split content and meta description
        content_parts = full_response_text.split("META_DESCRIPTION:")
        generated_content = content_parts[0].strip()
        
        meta_description = ""
        if len(content_parts) > 1:
            meta_description = content_parts[1].strip()
        else:
            st.warning(f"Meta description separator not found for topic '{topic_title}'. Will attempt to generate one separately or use a default.")
            # Fallback: could try another LLM call just for meta description if critical
            meta_prompt = f"Generate a concise and compelling meta description (150-160 characters) for a blog post titled: '{topic_title}'"
            meta_response = model.generate_content(meta_prompt)
            meta_description = meta_response.text.strip()


        if not generated_content:
            st.warning(f"Gemini returned no main content for topic: {topic_title}")
            return None, meta_description # Return meta_description even if content fails

        return generated_content, meta_description

    except Exception as e:
        st.error(f"Error during Gemini content generation for '{topic_title}': {e}")
        return None, None

def perform_internal_linking(all_posts_data, similarity_threshold=0.7, links_per_post=3, internal_link_density_target=0.075):
    """
    Analyzes generated posts and adds internal linking suggestions.

    Args:
        all_posts_data (list of dict): List of post data dictionaries. 
                                      Each dict needs 'title' and 'raw_content_md'.
                                      It will be updated with 'internal_links_to_add'.
        similarity_threshold (float): Threshold for considering posts similar for linking.
        links_per_post (int): Target number of internal links to add to each post.
        internal_link_density_target (float): Target for link density (not fully implemented yet).

    Returns:
        tuple: (updated_all_posts_data, linking_graph_img_bytes)
               linking_graph_img_bytes can be None if graph generation fails.
    """
    if not all_posts_data or len(all_posts_data) < 2:
        st.info("Not enough posts to perform internal linking.")
        return all_posts_data, None

    st.info(f"Starting internal linking analysis for {len(all_posts_data)} posts...")

    # 1. Get embeddings for all post titles (or full content for more accuracy, but slower)
    # For simplicity and speed, let's use titles first.
    # A better approach would be to embed chunks of content from each post.
    post_titles = [post['title'] for post in all_posts_data]
    with st.spinner("Generating embeddings for all post titles..."):
        title_embeddings = get_gemini_embeddings(post_titles)

    if title_embeddings is None or title_embeddings.size == 0:
        st.warning("Could not generate embeddings for post titles. Skipping internal linking.")
        return all_posts_data, None

    # Initialize 'internal_links_to_add' for all posts
    for post in all_posts_data:
        post['internal_links_to_add'] = [] # List of dicts: {target_post_index: X, anchor_text: "text", source_chunk: "chunk"}

    # 2. Compare each post with every other post
    # This is a simplified approach. A more robust one would use content chunks.
    # The current `compare_and_get_similar_chunks` and `get_keyword_suggestions_from_gemini`
    # are designed for one source -> one target. We need to adapt.

    # For now, let's simulate finding link opportunities and anchor text.
    # This section needs significant enhancement.
    # We'll use a placeholder logic: for each post, find a few other posts and try to link.
    
    # Placeholder: For each post (source_post), iterate through all other posts (target_post)
    for i, source_post_data in enumerate(all_posts_data):
        if len(source_post_data['internal_links_to_add']) >= links_per_post:
            continue # Already has enough links

        # Get chunks from the source post's content
        source_content_chunks = chunk_text_by_sentences(source_post_data['raw_content_md'])
        if not source_content_chunks:
            continue

        # Create a list of potential target posts (excluding the source post itself)
        potential_targets = [(idx, p) for idx, p in enumerate(all_posts_data) if idx != i]
        
        # Sort potential targets by some metric (e.g., similarity of titles - simplified)
        # This is where title_embeddings could be used for a quick sort.
        # For a more robust approach, compare source_content_chunks with target_post content/title.

        links_added_to_this_post = 0
        for target_idx, target_post_data in potential_targets:
            if links_added_to_this_post >= links_per_post:
                break

            # Simulate finding a relevant chunk and anchor text using a simplified Gemini call
            # This reuses get_keyword_suggestions_from_gemini logic but needs careful adaptation
            # For now, we'll pick a chunk and ask Gemini for anchor text related to target_post_data['title']
            
            # Create a mock similar_chunks_dict for the current source_post_chunks and target_post_data['title']
            # This is a placeholder for a more sophisticated semantic search
            mock_similar_chunks = {chunk: 0.8 for chunk in source_content_chunks[:5]} # Take first 5 chunks as candidates

            if not mock_similar_chunks:
                continue

            # st.spinner(f"Finding anchor text in '{source_post_data['title']}' for '{target_post_data['title']}'...")
            # The original get_keyword_suggestions_from_gemini shows a progress bar, which might be too much here.
            # We might need a version without Streamlit elements for batch processing.
            # For now, let's assume we can get a suggestion.
            
            # Simplified anchor text generation (placeholder)
            # In a real scenario, we'd call Gemini here for each pair.
            # This can be very slow for N*M calls.
            # A better way: batch prompts or a more advanced linking model.
            
            suggested_anchor = f"learn more about {target_post_data['title'][:30]}..." # Placeholder anchor
            found_chunk_for_anchor = source_content_chunks[0] # Placeholder chunk

            # Check if this link already exists (to avoid duplicate links to the same target from the same source)
            already_linked = any(link_info['target_post_index'] == target_idx for link_info in source_post_data['internal_links_to_add'])
            if not already_linked and suggested_anchor != "None":
                source_post_data['internal_links_to_add'].append({
                    "target_post_index": target_idx, # Store index of target post
                    "target_post_title": target_post_data['title'],
                    "anchor_text_suggestion": suggested_anchor,
                    "source_chunk_context": found_chunk_for_anchor 
                })
                links_added_to_this_post += 1
                # st.write(f"Link from {source_post_data['title']} to {target_post_data['title']} with anchor '{suggested_anchor}'")


    # 3. (Optional) Generate linking map using NetworkX
    # This is a very basic graph visualization.
    linking_graph_img_bytes = None
    try:
        G = nx.DiGraph()
        for i, post in enumerate(all_posts_data):
            G.add_node(i, label=post['title'][:20]+"...") # Use post index as node ID
            for link_info in post.get('internal_links_to_add', []):
                target_idx = link_info['target_post_index']
                if G.has_node(target_idx): # Ensure target node exists
                     G.add_edge(i, target_idx)
        
        if G.number_of_nodes() > 0:
            import io
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=0.5, iterations=50) # k adjusts distance between nodes
            labels = nx.get_node_attributes(G, 'label')
            nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, node_color="skyblue", 
                    font_size=8, font_weight="bold", arrowsize=15)
            plt.title("Internal Linking Map (Simplified)")
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150)
            img_buffer.seek(0)
            linking_graph_img_bytes = img_buffer.getvalue()
            plt.clf() # Clear the figure
            st.image(linking_graph_img_bytes, caption="Internal Linking Map Preview")
    except Exception as e:
        st.warning(f"Could not generate linking map: {e}")
        linking_graph_img_bytes = None

    st.success(f"Internal linking analysis complete. Processed {len(all_posts_data)} posts.")
    return all_posts_data, linking_graph_img_bytes

def create_output_zip(all_posts_data, linking_map_bytes=None):
    """
    Creates a ZIP file containing HTML posts, metadata CSV, linking map, and instructions.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_f:
        # 1. Add HTML files
        for i, post_data in enumerate(all_posts_data):
            post_title_slug = re.sub(r'\W+', '-', post_data['title'].lower())
            html_filename = f"wordpress_posts/post_{i+1:03d}_{post_title_slug[:50]}.html"
            # Ensure wordpress_html is generated if not already
            if not post_data.get("wordpress_html"):
                 # Basic conversion if not done, ideally this is already populated with links
                html_content = markdown.markdown(post_data.get('raw_content_md', ''))
                post_data["wordpress_html"] = f"<h1>{post_data['title']}</h1>\n{html_content}"
            
            zip_f.writestr(html_filename, post_data["wordpress_html"])

        # 2. Create and add metadata.csv
        metadata_output = io.StringIO()
        csv_writer = csv.writer(metadata_output)
        # Define headers for CSV
        headers = ["id", "title", "meta_description", "category", "tags", "filename"]
        csv_writer.writerow(headers)
        for i, post_data in enumerate(all_posts_data):
            post_title_slug = re.sub(r'\W+', '-', post_data['title'].lower())
            html_filename = f"post_{i+1:03d}_{post_title_slug[:50]}.html"
            tags_str = ", ".join(post_data.get("tags", []))
            csv_writer.writerow([
                f"post_{i+1:03d}",
                post_data.get("title", ""),
                post_data.get("meta_description", ""),
                post_data.get("category", "General"),
                tags_str,
                html_filename
            ])
        zip_f.writestr("wordpress_posts/metadata.csv", metadata_output.getvalue())
        metadata_output.close()

        # 3. Add linking_map.png (if available)
        if linking_map_bytes:
            zip_f.writestr("wordpress_posts/linking_map.png", linking_map_bytes)

        # 4. Add import_instructions.txt
        instructions = """
        WordPress Import Instructions:

        Method 1: Manual Copy-Paste
        1. For each .html file in the 'wordpress_posts' directory:
           a. Open the .html file in a web browser or text editor.
           b. Select and copy the entire content.
           c. In your WordPress admin, go to Posts > Add New.
           d. Enter the post title (from metadata.csv or filename).
           e. Switch the editor to 'Text' or 'Code editor' mode.
           f. Paste the copied HTML content.
           g. Assign categories and tags as per metadata.csv.
           h. Set a featured image (suggestions might be part of a future version).
           i. Save as draft or publish.

        Method 2: Using a WordPress HTML Import Plugin
        1. Search for and install a plugin like "HTML Import 2" or "WP All Import" (with its HTML import add-on).
        2. Configure the plugin to import the .html files from the 'wordpress_posts' directory.
        3. You may be able to map metadata from the 'metadata.csv' file if the plugin supports CSV for metadata population during HTML import.
           Refer to the specific plugin's documentation.
        4. Run the import process.

        Notes:
        - Review each post after import for formatting and ensure links work as expected.
        - The 'linking_map.png' provides a visual overview of internal links created.
        - This is a basic export. For more advanced features like scheduling, featured images, etc., direct WordPress API integration would be needed (future enhancement).
        """
        zip_f.writestr("wordpress_posts/import_instructions.txt", instructions)

    zip_buffer.seek(0)
    return zip_buffer

# --- Streamlit UI ---

st.set_page_config(page_title="SEO Blog Post Generator", layout="wide")

st.title("🤖 AI-Powered SEO Blog Post Generator & Interlinker")
st.markdown("""
This application leverages AI to generate unique, SEO-optimized blog posts for your website,
complete with automatic internal linking and WordPress-ready export.
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
st.header("📝 Input Parameters")

website_url = st.text_input("Target Website URL (for analysis):", 
                            placeholder="e.g., https://example.com")

num_posts = st.number_input("Number of Posts to Generate:", min_value=1, max_value=1000, value=100) # Max 1000 for sanity

brand_voice_options = ["formal", "casual", "technical", "friendly", "expert"]
brand_voice = st.selectbox("Brand Voice Preference:", brand_voice_options, index=1) # Default to casual

st.subheader("Advanced Options")
with st.expander("Configure Advanced Settings (Optional)"):
    target_keywords_csv = st.file_uploader("Upload Target Keywords CSV (Optional):", type=['csv'])
    # Placeholder for more advanced options from the spec
    # e.g., content length, linking density, style guide, custom prompts
    min_word_count = st.number_input("Minimum Word Count per Post:", min_value=300, value=800, step=50)
    max_word_count = st.number_input("Maximum Word Count per Post:", min_value=500, value=1200, step=50)
    internal_link_density = st.slider("Target Internal Link Density (% of content):", 
                                      min_value=1.0, max_value=20.0, value=7.5, step=0.5,
                                      help="Approximate percentage of content that should be internal links.")


if st.button("🚀 Generate Blog Posts", type="primary", use_container_width=True):
    if not gemini_api_key:
        st.error("🚨 Please enter your Gemini API Key in the sidebar to proceed.")
    elif not website_url:
        st.warning("⚠️ Please enter the Target Website URL.")
    elif not (website_url.startswith("http://") or website_url.startswith("https://")):
        st.warning("⚠️ Please enter a valid Website URL (starting with http:// or https://).")
    else:
        if not configure_gemini(gemini_api_key):
            st.stop() 

        st.info(f"Starting blog post generation for: {website_url}")
        st.info(f"Number of posts: {num_posts}, Brand voice: {brand_voice}")
        
        user_keywords_df = None
        if target_keywords_csv:
            try:
                user_keywords_df = pd.read_csv(target_keywords_csv)
                st.info(f"Successfully loaded keywords from uploaded CSV: {target_keywords_csv.name}")
                with st.expander("View Uploaded Keywords"):
                    st.dataframe(user_keywords_df)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                user_keywords_df = None
        
        # Placeholder for the main generation logic
        # This will involve:
        # 1. Website Analysis (scrape, extract themes/keywords)
        # 2. Topic Generation
        # 3. Content Generation for each topic
        # 4. Internal Linking
        # 5. Output Preparation

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # --- 1. Website Analysis ---
            status_text.text(f"Fetching content from {website_url}...")
            site_title, site_content = fetch_blog_content(website_url) # Re-using existing function

            if not site_content:
                st.error(f"Could not fetch content from {website_url}. Please check the URL or website structure.")
                st.stop()
            
            status_text.text("Analyzing website content with AI...")
            analyzed_data = analyze_website_with_gemini(site_content, brand_voice)

            if not analyzed_data:
                st.error("Failed to analyze website content. Please check logs or try again.")
                st.stop()
            
            st.success(f"Website analysis complete. Main Topic: {analyzed_data.get('main_topic', 'N/A')}")
            with st.expander("View Full Website Analysis Details"):
                st.json(analyzed_data)
            progress_bar.progress(10)

            # --- 2. Topic Generation ---
            status_text.text("Generating blog topics with AI...")
            generated_topics = generate_blog_topics_with_gemini(
                analyzed_data, 
                num_posts, 
                brand_voice, 
                user_keywords_df=user_keywords_df
            )

            if generated_topics is None: # Indicates an error occurred in generation
                st.error("Failed to generate blog topics. Please check logs or API key and try again.")
                st.stop()
            if not generated_topics: # Empty list but no hard error
                st.warning("No blog topics were generated. The AI might need more specific input or the website content was too generic.")
                st.stop()

            st.success(f"Successfully generated {len(generated_topics)} blog topics.")
            with st.expander("View Generated Topics"):
                for t in generated_topics:
                    st.write(t)
            progress_bar.progress(20)

            all_posts_content = [] # To store {title: "...", content: "...", internal_links: []}

            # --- 3. Content Generation ---
            successful_posts_generated = 0
            for i, topic_title in enumerate(generated_topics):
                status_text.text(f"Processing topic: {topic_title} ({i+1}/{len(generated_topics)})") # Simplified f-string
                
                # Consolidate keywords for this specific post generation
                current_post_keywords = analyzed_data.get("keywords", [])
                if user_keywords_df is not None and not user_keywords_df.empty:
                    user_kw_list = []
                    for col in user_keywords_df.columns:
                        user_kw_list.extend(user_keywords_df[col].dropna().astype(str).tolist())
                    current_post_keywords = list(set(current_post_keywords + user_kw_list))

                generated_article_md, meta_desc = generate_post_content_with_gemini(
                    topic_title,
                    analyzed_data,
                    brand_voice,
                    min_word_count,
                    max_word_count,
                    existing_keywords=current_post_keywords
                )

                if generated_article_md:
                    all_posts_content.append({
                        "title": topic_title,
                        "raw_content_md": generated_article_md, # Store Markdown content
                        "meta_description": meta_desc if meta_desc else f"Read our latest article on {topic_title}.",
                        "category": analyzed_data.get("main_topic", "General"),
                        "tags": current_post_keywords[:5], # Take first 5 keywords as tags for now
                        "wordpress_html": "", # To be filled after linking
                        "internal_links_to_add": [] # To be filled by linking logic
                    })
                    successful_posts_generated +=1
                else:
                    st.warning(f"Skipping post for topic '{topic_title}' due to content generation failure.")

                progress_bar.progress(20 + int(60 * (i+1)/len(generated_topics)) )
            
            if successful_posts_generated == 0 and len(generated_topics) > 0:
                st.error("No posts were successfully generated. Please check API key, model access, or Gemini service status.")
                st.stop()
                
            st.success(f"Successfully generated content for {successful_posts_generated} posts.")

            # --- 4. Internal Linking ---
            status_text.text("Performing internal linking analysis...")
            
            # The perform_internal_linking function needs to be more robust.
            # The current placeholder is very basic.
            # It should use embeddings for similarity and a better way to suggest anchors.
            all_posts_content, linking_map_bytes = perform_internal_linking(
                all_posts_content, 
                similarity_threshold=0.65, # Example threshold
                links_per_post=st.session_state.get("advanced_links_per_post", 3) # Get from advanced options if set
            )
            # Store linking_map_bytes in session state if needed for download
            if linking_map_bytes:
                st.session_state['linking_map_bytes'] = linking_map_bytes
            
            progress_bar.progress(90)

            # --- 5. Output Preparation & WordPress Formatting ---
            status_text.text("Preparing outputs and WordPress format...")
            # TODO: Convert raw_content + internal_links into WordPress HTML
            # TODO: Create metadata.csv, linking_map.png, import_instructions.txt
            # TODO: Zip files for download
            
            # Placeholder for HTML generation & inserting links
            for post_idx, post_data in enumerate(all_posts_content):
                final_html_content = markdown.markdown(post_data['raw_content_md']) # Convert base MD to HTML

                # Attempt to insert internal links
                # This is a very naive insertion and needs to be much smarter
                # e.g., finding the source_chunk_context in the HTML and replacing/wrapping anchor text
                if post_data.get('internal_links_to_add'):
                    for link_info in post_data['internal_links_to_add']:
                        target_post_title_slug = re.sub(r'\W+', '-', link_info['target_post_title'].lower()) + ".html"
                        # Naive: try to replace suggested anchor text if found.
                        # A robust solution would find the source_chunk_context and then the anchor within it.
                        anchor_text = link_info['anchor_text_suggestion']
                        if anchor_text in final_html_content:
                             final_html_content = final_html_content.replace(
                                 anchor_text, 
                                 f'<a href="./{target_post_title_slug}">{anchor_text}</a>', 
                                 1 # Replace only first occurrence for now
                             )
                        # else:
                            # st.warning(f"Could not find anchor '{anchor_text}' in post '{post_data['title']}' to link to '{link_info['target_post_title']}'")
                
                post_data["wordpress_html"] = f"<h1>{post_data['title']}</h1>\n{final_html_content}"

            time.sleep(1) # Simulate work
            st.success("Outputs formatted (basic HTML with attempted links).")
            progress_bar.progress(100)
            
            # --- Display Preview & Download Options ---
            st.subheader("🎉 Generation Complete!")
            
            # Preview (first 3 posts)
            st.markdown("### Preview of Generated Posts (First 3)")
            for i, post_data in enumerate(all_posts_content[:3]):
                with st.expander(f"Post: {post_data['title']}"):
                    st.markdown(f"**Meta Description:** {post_data['meta_description']}")
                    st.markdown("**Content (Markdown Source):**")
                    st.markdown(f"```markdown\n{post_data['raw_content_md'][:1000]}...\n```") # Preview Markdown
                    # st.code(post_data['wordpress_html'], language='html') # Keep HTML preview if needed later
            
            # --- Download Button ---
            if all_posts_content:
                st.markdown("---")
                st.subheader("📥 Download Generated Content")
                
                zip_buffer = create_output_zip(
                    all_posts_content, 
                    st.session_state.get('linking_map_bytes')
                )
                
                st.download_button(
                    label="Download All Posts as ZIP",
                    data=zip_buffer,
                    file_name=f"{re.sub(r'[^a-zA-Z0-9]+', '-', website_url.replace('https://','').replace('http://',''))}_posts.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.warning("No content was generated to download.")


            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            status_text.empty()
            progress_bar.empty()


st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini. For educational and demonstration purposes.")
st.markdown("Considerations: Web page structures vary. Content extraction might not be perfect for all sites.")
