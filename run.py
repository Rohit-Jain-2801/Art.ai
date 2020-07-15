# Importing libraries & components
from flask import Flask, render_template

# Flask app initialization
app = Flask(__name__)

# routings
@app.route("/")
def home():
    return render_template('index.html')

# running the server
if __name__ == '__main__':
    app.run(debug=True)