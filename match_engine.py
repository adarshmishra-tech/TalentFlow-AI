import os
import logging
from typing import List, Dict
import concurrent.futures
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import PyPDF2
import docx

# Assume utils.text_cleaner provides a clean_text function
from utils.text_cleaner import clean_text

# Set up logging for the module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_resume(path: str) -> str:
    """
    Parse the content of a resume file (PDF or DOCX) and extract raw text.

    :param path: Path to the resume file.
    :return: Extracted text from the file.
    :raises ValueError: If the file type is unsupported.
    :raises Exception: If file reading fails.
    """
    if path.lower().endswith('.pdf'):
        # Parse PDF using PyPDF2
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ' '.join(page.extract_text() for page in reader.pages if page.extract_text())
    elif path.lower().endswith('.docx'):
        # Parse DOCX using python-docx
        doc = docx.Document(path)
        text = ' '.join(para.text for para in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return text


def get_top_matches(job_desc: str, resume_paths: List[str]) -> List[Dict[str, any]]:
    """
    Compute semantic similarity between the job description and resumes, rank them,
    and return the top 10 matches with scores and keyword highlights.

    :param job_desc: Job or project description as a string.
    :param resume_paths: List of file paths to resumes (PDF or DOCX).
    :return: List of dictionaries with 'filename', 'score', and 'highlights' for top 10 matches.
    :raises ValueError: If inputs are empty or invalid, or no valid resumes can be parsed.
    """
    if not job_desc.strip():
        raise ValueError("Job description cannot be empty.")
    if not resume_paths:
        raise ValueError("At least one resume path is required.")

    logger.info(f"Processing {len(resume_paths)} resumes for matching.")

    # Parse resumes in parallel using multithreading for efficiency
    resume_texts: List[str] = []
    filenames: List[str] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(parse_resume, path): path for path in resume_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                raw_text = future.result()
                cleaned_text = clean_text(raw_text)
                if cleaned_text.strip():  # Skip empty cleaned texts
                    resume_texts.append(cleaned_text)
                    filenames.append(os.path.basename(path))
            except Exception as e:
                logger.warning(f"Failed to parse or clean {path}: {e}")

    if not resume_texts:
        raise ValueError("No valid resumes could be parsed. Check file formats and contents.")

    logger.info(f"Successfully parsed {len(resume_texts)} valid resumes.")

    # Load semantic embedding model
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed job description
    job_embedding = model.encode(job_desc)

    # Embed all resume texts in batch for efficiency (handles 1000+ easily)
    resume_embeddings = model.encode(resume_texts, batch_size=32, show_progress_bar=False)

    # Compute cosine similarities
    similarities = cosine_similarity([job_embedding], resume_embeddings)[0]

    # Prepare TF-IDF for keyword extraction and highlights
    logger.info("Computing TF-IDF for keyword highlights...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features for scalability
    all_texts = [job_desc] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Extract top keywords from job description (top 20 for broader matching)
    job_tfidf = tfidf_matrix[0]
    if job_tfidf.nnz == 0:
        logger.warning("No keywords extracted from job description.")
        job_keywords = []
    else:
        sorted_indices = job_tfidf.indices[np.argsort(job_tfidf.data)[-20:]]
        job_keywords = [feature_names[i] for i in sorted_indices[::-1]]  # Descending order

    # Build results list
    results: List[Dict[str, any]] = []
    for i in range(len(resume_texts)):
        score = similarities[i]

        # Get matching highlights for this resume
        resume_tfidf = tfidf_matrix[i + 1]
        resume_features = set(feature_names[resume_tfidf.indices])
        matches = [kw for kw in job_keywords if kw in resume_features]
        highlights = ', '.join(matches[:10])  # Limit to top 10 matches

        results.append({
            'filename': filenames[i],
            'score': round(float(score), 4),  # Round for clean output
            'highlights': highlights
        })

    # Sort by score descending and return top 10
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    logger.info("Matching completed successfully.")
    return sorted_results[:10]
