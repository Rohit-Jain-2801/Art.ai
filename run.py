# import eventlet
# eventlet.monkey_patch()

# Importing components
from nst import app, socketio

# running the server
if __name__ == '__main__':
    # socketio.run(app=app, debug=False, host='0.0.0.0')
    socketio.run(app=app, debug=False)