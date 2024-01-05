import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import os
absolute_path = os.path.abspath(os.path.dirname(__file__))
pickle_file_path = os.path.join(absolute_path, 'food_dict.pkl')




import pickle
foods_dict = pickle.load(open(pickle_file_path, 'rb'))

foods=pd.DataFrame(foods_dict)
cosine_sim_path = os.path.join(absolute_path, 'cosin_sim.pkl')
cosine_sim = pickle.load(open(cosine_sim_path, 'rb'))
rating_matrix_path = os.path.join(absolute_path, 'rating_matrix1.pkl')
user_rating_path = os.path.join(absolute_path, 'user_rating.pkl')
rx = pickle.load(open(rating_matrix_path, 'rb'))
ux = pickle.load(open(user_rating_path, 'rb'))
rating_matrix=pd.DataFrame(rx)
userratingdf=pd.DataFrame(ux)
# Just considering the Food names from the dataframe
indices = pd.Series(foods.index, index=foods['Name']).drop_duplicates()

# Assuming 'rating_matrix' is your original DataFrame
current_size = len(rating_matrix)
rows_to_add = 400 - current_size

# Check if rows need to be added
if rows_to_add > 0:
    # Assuming your DataFrame columns are labeled as in your example (1.0, 2.0, ..., 100.0)
    columns = rating_matrix.columns
    # Create a DataFrame with additional rows filled with zeros (or NaN if preferred)
    additional_rows = pd.DataFrame(np.zeros((rows_to_add, len(columns))), columns=columns)
    # Use concat to combine the original DataFrame with the additional rows
    rating_matrix = pd.concat([rating_matrix, additional_rows], ignore_index=True)

rating_matrix.fillna(0, inplace=True)




# Adjusting the function to recommend dishes that do not contain a specified ingredient
# This can be useful for people with allergies or dietary restrictions

def recommend_by_that_ingredient(ingredient, df):
    """
    Recommend dishes that do not contain the specified ingredient.

    :param ingredient: The ingredient to avoid.
    :param df: DataFrame containing the food data.
    :return: DataFrame with dishes that do not contain the ingredient.
    """
    relevant_dishes = df[df['Describe'].str.contains(ingredient, case=False, na=False)]
    return relevant_dishes[['Name', 'Describe']].head(3)





# Dietary Restrictions: Cater to different dietary needs such as vegetarian, vegan, gluten-free, dairy-free, nut-free, halal, or koshe

# The main recommender code CONTENT BASED!
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar food
    sim_scores = sim_scores[1:6]

    food_indices = [i[0] for i in sim_scores]
    return foods['Name'].iloc[food_indices]
    #return foods['Name'].iloc[food_indices]

#csr_rating_matrix =  csr_matrix(rating_matrix.values)
# Using cosine similarity to find nearest neigbours
#recommender = NearestNeighbors(metric='cosine')
#recommender.fit(csr_rating_matrix)

# The main recommender code COLLABORATION BASED!
#def Get_Recommendations(title):
   # user = foods[foods['Name'] == title]
    #user_index = np.where(rating_matrix.index == int(user['Food_ID']))[0][0]
    #user_ratings = rating_matrix.iloc[user_index]

    #reshaped = user_ratings.values.reshape(1, -1)
    #distances, indices = recommender.kneighbors(reshaped, n_neighbors=16)

    #nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
    #nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})

    #result = pd.merge(nearest_neighbors, foods, on='Food_ID', how='left')

    #return result.head()


#option1 = st.selectbox(
  #  'Provide the food name and we will provide to u similar users based foods',
  #  foods['Name'].values)
#if st.button('Recommend using collaboration filtering'):
   # recommendation1=Get_Recommendations(option1)
   # st.write('You selected:',recommendation1)
# Testing the function with an example ingredient to avoid
# For example, let's avoid "chicken" as an ingredient







# Create dummy features based on ingredient analysis
foods['high_protein'] = foods['Describe'].apply(lambda x: 1 if 'chicken' in x or 'beef' in x or 'lentils' in x else 0)
foods['high_carb'] = foods['Describe'].apply(lambda x: 1 if 'bread' in x or 'rice' in x or 'pasta' in x else 0)

# Select features and labels
X = foods[['high_protein', 'high_carb']]  # These are dummy features for demonstration
y_protein = foods['high_protein']
y_carb = foods['high_carb']


# Assuming that we have already identified which foods are high in protein and carbs
# and those columns are named 'high_protein' and 'high_carb' respectively
# For the sake of this example, let's create dummy data for these columns


# Function to suggest top 5 foods based on protein and carb combination
def suggest_foods(df, protein_status, carb_status):
    """
    Suggest top 5 foods based on given protein and carbohydrate status.

    :param df: DataFrame containing the food data.
    :param protein_status: 'high' or 'low' protein.
    :param carb_status: 'high' or 'low' carbohydrate.
    :return: DataFrame with top 5 food suggestions.
    """
    if protein_status == 'high':
        protein_filter = df['high_protein'] == 1
    else:
        protein_filter = df['high_protein'] == 0

    if carb_status == 'high':
        carb_filter = df['high_carb'] == 1
    else:
        carb_filter = df['high_carb'] == 0

    filtered_df = df[protein_filter & carb_filter]
    return filtered_df[['Name']].head(5)



from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# MongoDB Atlas connection URI
uri = "mongodb+srv://kzaidnba:EuL9aQQpoN35onPO@cluster0.m6hepyg.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    db = client['zaidkrecommender']
    collection = db['recc']
    cursor = collection.find({})
    data_list = list(cursor)
    data_2 = pd.DataFrame(data_list)

    # Drop the '_id' column as it's not needed for analysis
    if '_id' in data_2.columns:
        data_2.drop('_id', axis=1, inplace=True)

    # Convert 'User_ID', 'Food_ID', and 'Rating' to numeric types
    data_2['User_ID'] = pd.to_numeric(data_2['User_ID'], errors='coerce')
    data_2['Food_ID'] = pd.to_numeric(data_2['Food_ID'], errors='coerce')
    data_2['Rating'] = pd.to_numeric(data_2['Rating'], errors='coerce')

    # Drop any rows with NaN values (if they exist after conversion)
    data_2.dropna(subset=['User_ID', 'Food_ID', 'Rating'], inplace=True)

except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


import os

# Get the absolute path to the directory where your app.py is located
absolute_path = os.path.abspath(os.path.dirname(__file__))

# Define the path to your pickle file using an absolute path
food_dict_collab_path = os.path.join(absolute_path, 'food_dictfinalcollab.pkl')

import pickle
import pandas as pd

# Load the pickle file using the absolute path
foods_dict4 = pickle.load(open(food_dict_collab_path, 'rb'))

# Convert foods_dict4 to a DataFrame
food_info_df = pd.DataFrame(foods_dict4)


from pymongo import UpdateOne

from pymongo import UpdateOne

from pymongo import UpdateOne

def update_mongodb_ratings(db, updated_ratings_df):
    update_operations = []
    for index, row in updated_ratings_df.iterrows():
        filter_query = {'User_ID': row['User_ID'], 'Food_ID': row['Food_ID']}
        new_values = {"$set": {'Rating': row['Rating']}}
        update_op = UpdateOne(filter_query, new_values, upsert=True)  # Notice the upsert=True here
        update_operations.append(update_op)

    try:
        db.bulk_write(update_operations)
    except Exception as e:
        print("An error occurred while updating MongoDB:", e)




# Function to recommend food items using collaborative filtering
def recommend_food_items(user_id, num_recommendations, predictions_df, original_ratings_df, food_info_df):
    # Ensure 'Food_ID' is a column in original_ratings_df
    if 'Food_ID' not in original_ratings_df.columns:
        original_ratings_df = original_ratings_df.reset_index()

    # Get and sort the user's predictions
    sorted_user_predictions = predictions_df.loc[user_id].sort_values(ascending=False)

    # Get the user's data
    user_data = original_ratings_df[original_ratings_df.User_ID == user_id]

    # Recommend the highest predicted rating foods that the user hasn't seen yet
    recommendations = (sorted_user_predictions[~sorted_user_predictions.index.isin(user_data['Food_ID'])].
                       sort_values(ascending=False).
                       reset_index().rename(columns={user_id: 'PredictedRating'}).
                       head(num_recommendations))

    # Merge with the food_info_df to get food names
    recommendations = recommendations.merge(food_info_df, how='left', on='Food_ID')

    return recommendations[['Name', 'PredictedRating']]



# Load your data (replace 'path_to_file' with your file path)


# Aggregating the ratings by taking the mean for each user-food pair
agg_ratings = data_2.groupby(['User_ID', 'Food_ID']).mean().reset_index()

# Creating the pivot table
pivot_table = agg_ratings.pivot(index='User_ID', columns='Food_ID', values='Rating').fillna(0)
matrix = pivot_table.values

# Normalize by subtracting the mean rating for each user
user_ratings_mean = np.mean(matrix, axis = 1)
matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

# Singular Value Decomposition
U, sigma, Vt = svds(matrix_demeaned, k = 50)
sigma = np.diag(sigma)

# Making predictions from the decomposed matrices
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# Convert the reconstructed matrix back to a DataFrame
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_table.columns, index = pivot_table.index)


import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

def update_user_rating(user_id, food_id, new_rating, ratings_df, cf_model_k):
    """
    Updates the rating for a user, retrains the collaborative filtering model, and returns updated predictions.

    Args:
    user_id (int): The ID of the user.
    food_id (int): The ID of the food item.
    new_rating (float): The new rating value.
    ratings_df (pd.DataFrame): The DataFrame containing all user ratings.
    cf_model_k (int): The number of latent factors for the matrix factorization.

    Returns:
    pd.DataFrame: The updated predictions DataFrame after retraining the model.
    """

    # Updating or adding the new rating
    if ratings_df[(ratings_df.User_ID == user_id) & (ratings_df.Food_ID == food_id)].empty:
        # Add a new rating if it doesn't exist
        new_row = pd.DataFrame({'User_ID': [user_id], 'Food_ID': [food_id], 'Rating': [new_rating]})
        ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    else:
        # Update the rating if it already exists
        ratings_df.loc[(ratings_df.User_ID == user_id) & (ratings_df.Food_ID == food_id), 'Rating'] = new_rating

    # Creating the pivot table
    pivot_table = ratings_df.pivot_table(index='User_ID', columns='Food_ID', values='Rating', fill_value=0)

    # Matrix Factorization
    matrix = pivot_table.values
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(matrix_demeaned, k=cf_model_k)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    # Updated predictions
    updated_predictions_df = pd.DataFrame(all_user_predicted_ratings, columns=pivot_table.columns, index=pivot_table.index)

    return ratings_df, updated_predictions_df



# Initialize predictions_df with cf_preds_df
predictions_df = cf_preds_df
# Streamlit Interface
st.title('Food Item Recommender Engine-Zaid Khan')

# Content-based recommendation
st.subheader("Content-based Recommendation")
selected_food = st.selectbox('Select a food to find similar items:', foods['Name'].values)
if st.button('Recommend Similar Foods'):
    recommendations = get_recommendations(selected_food)
    st.write(recommendations)

# Recommendation by ingredient
st.subheader("Recommendation by Ingredient")
ingredient = st.text_input('Enter an ingredient to avoid:')
if st.button('Recommend Foods With Ingredient'):
    recommendations = recommend_by_that_ingredient(ingredient, foods)
    st.write(recommendations)

# Suggest foods based on protein and carb
st.subheader("Suggest Foods Based on Protein and Carb")
protein_status = st.selectbox('Select Protein Status:', ['high', 'low'])
carb_status = st.selectbox('Select Carb Status:', ['high', 'low'])
if st.button('Suggest Foods'):
    food_suggestions = suggest_foods(foods, protein_status, carb_status)
    st.write(food_suggestions)

# Collaborative filtering recommendation
st.subheader("Collaborative Filtering Recommendation")
user_id = st.number_input('Enter User ID:', min_value=1, step=1)
num_recommendations = st.slider('Number of Recommendations:', 1, 10, 5)
if st.button('Get Recommendations'):
    recommendations = recommend_food_items(user_id, num_recommendations, predictions_df, original_ratings_df=data_2, food_info_df=food_info_df)
    st.write(recommendations)

# Streamlit Interface for Updating User Ratings
st.subheader("Update User Rating")
update_user = st.number_input('User ID for Rating Update:', min_value=1, step=1)
update_food_name = st.selectbox('Select Food for Rating Update:', foods['Name'].values)
new_rating = st.slider('New Rating:', 0.0, 10.0, 3.5)

if st.button('Update Rating'):
    # Find the corresponding Food ID
    update_food_id = foods[foods['Name'] == update_food_name].index[0]+1

    # Update the rating using the Food ID
    data_2, updated_preds = update_user_rating(update_user, update_food_id, new_rating, ratings_df=data_2, cf_model_k=50)

    # Update predictions_df with the new predictions
    predictions_df = updated_preds

    # Call the function to update the MongoDB database
    update_mongodb_ratings(collection, data_2)

    st.success('Rating Updated!')

# Your link URL
url = 'https://www.linkedin.com/in/zaid-khan-903228238/'

# Your link text
link_text = 'Click here to visit my linkedin profile'

# Using markdown to create a hyperlink
st.markdown(f'[{link_text}]({url})')

    

