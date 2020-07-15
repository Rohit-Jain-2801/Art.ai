# Importing libraries
from flask import Flask
from flask_socketio import SocketIO

# Flask app initialization
app = Flask(import_name=__name__,)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app=app, always_connect=True)

from nst import routes