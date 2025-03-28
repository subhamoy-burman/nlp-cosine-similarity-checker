import nltk
import string

# These are the NEW imports from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """Cleans text and returns a single string of processed words."""
    # 1. Tokenize sentences, then words
    sentences = nltk.sent_tokenize(text)
    words = []
    for sent in sentences:
        words.extend(nltk.word_tokenize(sent))

    # 2. Lowercase
    words = [w.lower() for w in words]

    # 3. Remove Stopwords
    stopwords_list = nltk.corpus.stopwords.words('english')
    words = [w for w in words if w not in stopwords_list]

    # 4. Remove Punctuation
    words = [w for w in words if w not in string.punctuation]

    # 5. Keep only alphabetic tokens
    words = [w for w in words if w.isalpha()]

    # 6. IMPORTANT: Join back into a single string
    return " ".join(words)

def calculate_similarity(filepath1, filepath2):
    print(f"--- Calculating Similarity between '{filepath1}' and '{filepath2}' ---")
    # Step 4a: Read file contents
    with open(filepath1, 'r', encoding='utf-8') as f1:
        text1 = f1.read()
    with open(filepath2, 'r', encoding='utf-8') as f2:
        text2 = f2.read()

    # Step 4b: Preprocess the texts
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    print("\nProcessed Text 1:", text1_processed)
    print("Processed Text 2:", text2_processed)

    print("\n--- Using Bag-of-Words (BoW) ---")
    # Create the BoW vectorizer
    vectorizer = CountVectorizer()

    # Learn vocabulary and create word count vectors for BOTH documents at once
    # Input must be a list or iterable of strings
    documents = [text1_processed, text2_processed]
    bow_matrix = vectorizer.fit_transform(documents)

    # Print the vocabulary found (optional, for understanding)
    # print("Vocabulary:", vectorizer.get_feature_names_out())
    # Print the BoW matrix (optional, shows counts)
    # print("BoW Matrix:\n", bow_matrix.toarray()) # .toarray() converts sparse matrix to dense

    # Calculate Cosine Similarity between the two vectors
    # bow_matrix[0] is the vector for doc1, bow_matrix[1] is for doc2
    # cosine_similarity returns a matrix, [[1.0, sim_1_2], [sim_2_1, 1.0]]
    # We want the similarity between doc 1 and doc 2, which is at [0][1] or [1][0]
    bow_similarity = cosine_similarity(bow_matrix[0:1], bow_matrix[1:2])[0][0] # Efficient way for 2 vectors
    print(f"BoW Cosine Similarity: {bow_similarity:.4f}") # Format to 4 decimal places# We will add BoW and TF-IDF steps here...

    print("\n--- Using TF-IDF ---")
    # Create the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Learn vocabulary+IDF and create TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents) # Use same 'documents' list

    # Print the vocabulary found (optional, same as BoW usually)
    # print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
    # Print the TF-IDF matrix (optional, shows scores)
    # print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

    # Calculate Cosine Similarity between the TF-IDF vectors
    tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    print(f"TF-IDF Cosine Similarity: {tfidf_similarity:.4f}")

    # --- Main part of the script ---
if __name__ == "__main__":
    # Define the file paths (make sure they match your file names)
    file1 = 'doc1.txt'
    file2 = 'doc2.txt'

    # Call the function
    calculate_similarity(file1, file2)