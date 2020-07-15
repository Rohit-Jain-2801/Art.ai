# Importing components
from nst import app
from flask import render_template, request

# routings
@app.route("/", methods=['POST'])
def home():
    return render_template('index.html')