from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = pd.read_csv("recipes.csv")

# Vectorization setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        include = request.form.get('include_ingredients')
        exclude = request.form.get('exclude_ingredients', '')

        if not include or not include.strip():
            return render_template('index.html', error="Please enter ingredients.")

        query_vec = vectorizer.transform([include])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
        top_indices = cosine_sim[0].argsort()[::-1]

        matches = []
        for idx in top_indices:
            recipe = df.iloc[idx]
            score = cosine_sim[0][idx]

            if score < 0.1:
                continue

            if exclude:
                excluded = [e.strip().lower() for e in exclude.split(',')]
                if any(e in recipe['ingredients'].lower() for e in excluded):
                    continue

            matches.append({
                'id': idx,
                'image': recipe['image'],
                'title': recipe['title'],
                'rating': recipe['rating']
            })

        if not matches:
            return render_template('index.html', error="No matching recipes found.")

        return render_template('results.html', recipes=matches)

    else:
        # GET request, just show search form
        return render_template('index.html')


@app.route('/recipe/<int:recipe_id>')
def recipe_detail(recipe_id):
    recipe = df.iloc[recipe_id]
    return render_template('recipe_detail.html', recipe=recipe)

if __name__ == '__main__':
    app.run(debug=True)
