# Importing components
from nst import app, socketio
from flask import render_template, request

# routings
@app.route("/", methods=['POST', 'GET'])
def home():
    return render_template(template_name_or_list='index.html')

@socketio.on('data')
def handle_data(data):
    print(data)