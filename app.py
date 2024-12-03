from flask import Flask, render_template, request
import sqlite3
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

# Function to get the connection to the database
def get_db_connection():
    conn = sqlite3.connect('recipes.db')
    conn.row_factory = sqlite3.Row
    return conn

# Fetch recipe data including the image from Recipes and stats from RecipeStat
def get_recipes_and_stats():
    conn = get_db_connection()
    
    # Fetch data from RecipeAvailable view (recipe details)
    cursor = conn.execute('''SELECT recipe_id, title, ingredients, dietary_info FROM RecipeAvailable''')
    recipes_data = cursor.fetchall()

    # Fetch image data from Recipes table
    cursor = conn.execute('''SELECT recipe_id, image FROM Recipes''')
    image_data = cursor.fetchall()

    # Convert image data into a dictionary
    image_dict = {row['recipe_id']: row['image'] for row in image_data}

    # Merge the recipe data with image data
    merged_recipes = []
    for recipe in recipes_data:
        recipe_id = recipe['recipe_id']
        image_url = image_dict.get(recipe_id, '')
        merged_recipes.append({
            'recipe_id': recipe_id,
            'title': recipe['title'],
            'ingredients': recipe['ingredients'],
            'dietary_info': recipe['dietary_info'],
            'image': image_url  # Add image URL from the Recipes table
        })
    
    conn.close()
    return merged_recipes

# Fetch the RecipeStat data for ingredient_level and dietary_info_level
def get_recipe_stats():
    conn = get_db_connection()
    cursor = conn.execute('''SELECT recipe_id, ingredient_level, dietary_info_level FROM RecipeStat''')
    stats_data = cursor.fetchall()
    
    # Convert stats data into a dictionary
    stats_dict = {row['recipe_id']: {
        'ingredient_level': row['ingredient_level'],
        'dietary_info_level': row['dietary_info_level']
    } for row in stats_data}
    
    conn.close()
    return stats_dict

# Fetch ingredients, stats, and image data
def get_recommended_recipes(user_input):
    # Prepare the recipe data
    recipes = get_recipes_and_stats()
    stats = get_recipe_stats()
    
    # Initialize MultiLabelBinarizer and transform the list of ingredients
    mlb = MultiLabelBinarizer()
    recipe_ingredients = [recipe['ingredients'].split(', ') for recipe in recipes]
    feature_matrix = mlb.fit_transform(recipe_ingredients)
    
    # Initialize KNN model and fit it with the feature matrix
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(feature_matrix)

    # Transform the user's input ingredients into the same format (one-hot encoding)
    user_input_list = [user_input.split(', ')]
    user_input_matrix = mlb.transform(user_input_list)

    # Compute distances and indices of nearest neighbors
    distances, indices = knn.kneighbors(user_input_matrix)

    # Get the recommended recipes and their distances
    recommended_recipes = [recipes[i] for i in indices[0]]
    for i, recipe in enumerate(recommended_recipes):
        recipe_id = recipe['recipe_id']
        recipe['distance'] = distances[0][i]
        # Add ingredient_level and dietary_info_level from RecipeStat
        recipe['ingredient_level'] = stats.get(recipe_id, {}).get('ingredient_level', '')
        recipe['dietary_info_level'] = stats.get(recipe_id, {}).get('dietary_info_level', '')

    return recommended_recipes

# Route to display the recipes based on the input ingredients
@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    recipes = []
    if request.method == 'POST':
        user_input = request.form['ingredients']
        recipes = get_recommended_recipes(user_input)
    
    return render_template('index.html', recipes=recipes, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
