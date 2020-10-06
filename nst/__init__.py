# Importing libraries
import os
from flask import Flask
from flask_socketio import SocketIO

# Flask app initialization
app = Flask(import_name=__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = os.urandom(16)

# socketio = SocketIO(app=app, always_connect=True, ping_interval=1000, ping_timeout=120000)
socketio = SocketIO(app=app, manage_session=False, always_connect=True, ping_interval=10000, ping_timeout=5000, cors_allowed_origins="*")

from nst import routes