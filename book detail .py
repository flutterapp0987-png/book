# ====================================================
# Import libraries
# ====================================================
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ====================================================
# Load dataset (already provided in Colab)
# ====================================================
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")
users = pd.read_csv("users.csv")

# ====================================================
# Filter dataset
# Keep only users with >=200 ratings and books with >=100 ratings
# ====================================================
ratings_per_user = ratings.groupby("User-ID").size()
ratings = ratings[ratings["User-ID"].isin(ratings_per_user[ratings_per_user >= 200].index)]

ratings_per_book = ratings.groupby("ISBN").size()
ratings = ratings[ratings["ISBN"].isin(ratings_per_book[ratings_per_book >= 100].index)]

# ====================================================
# Create pivot table (User x Book ratings matrix)
# ====================================================
book_ratings = ratings.merge(books, on="ISBN")
user_book_matrix = book_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating").fillna(0)

# Convert to sparse matrix for efficiency
book_sparse = csr_matrix(user_book_matrix.values)

# ====================================================
# Train KNN model
# ====================================================
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(book_sparse)

# ====================================================
# Define Recommendation Function
# ====================================================
def get_recommends(book_title):
    # Get index of the book
    book_list = user_book_matrix.index.tolist()
    idx = book_list.index(book_title)
    
    # Find nearest neighbors (k=6 because first one will be the book itself)
    distances, indices = model.kneighbors(book_sparse[idx], n_neighbors=6)
    
    # Build recommendations
    recommendations = []
    for i in range(1, len(distances[0])):  # skip the first (the book itself)
        title = book_list[indices[0][i]]
        dist = distances[0][i]
        recommendations.append([title, float(dist)])
    
    return [book_title, recommendations]

# ====================================================
# Test the function
# ====================================================
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
