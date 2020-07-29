# Importing components
from nst import app, socketio, utils
from nst.models import hub_model, tf_model
from flask import render_template, url_for, request

socket_user = {}

# routings
@app.route("/", methods=['GET'])
def home():
    return render_template(template_name_or_list='index.html')


class SocketConnection:
    def __init__(self, req):
        self.namespace = req.namespace
        self.id = req.sid
        self.connection = True
        print('Connected! ' + self.id)


    def disconnect(self):
        self.connection = False
        del socket_user[self.id]
        print('Disconnected! ' + self.id)


    def acknowledge(self):
        print('Data sent! ' + self.id)


    def handle_data(self, data):
        self.data = data
        self.content_img = utils.base_to_img(data=self.data['image1'])
        self.style_img = utils.base_to_img(data=self.data['image2'])
        
        if 'cfg' in self.data.keys():
            for self.epoch, self.update, self.output_img in tf_model.run_style_transfer(
                    content_img=self.content_img, 
                    style_img=self.style_img,
                    num_iterations=int(self.data['cfg']['epochs']),
                    content_weight=float(self.data['cfg']['cwt']),
                    style_weight=float(self.data['cfg']['swt']),
                    learning_rate=float(self.data['cfg']['lr'])
                ):
                
                if self.update:
                    self.output = utils.img_to_baseuri(img=self.output_img)
                    self.out = {
                        'epoch': self.epoch,
                        'output': self.output
                    }
                else:
                    self.out = {
                        'epoch': self.epoch,
                    }
                
                if self.connection:
                    socketio.send(data=self.out, json=False, namespace=self.namespace, room=self.id, callback=self.acknowledge, include_self=True)
                    socketio.sleep(1)
                else:
                    print('Stopping model call!')
                    break

        else:
            self.output_img = hub_model.run_style_transfer(content_img=self.content_img, style_img=self.style_img)
            self.output = utils.img_to_baseuri(img=self.output_img)
            socketio.emit(event='hub', data=str(self.output), json=False, namespace=self.namespace, room=self.id, callback=self.acknowledge, include_self=True)


@socketio.on('connect')
def connected():
    socket_user[request.sid] = SocketConnection(req=request)


@socketio.on('disconnect')
def disconnect():
    socket_user[request.sid].disconnect()


@socketio.on('data')
def handle_data(data):
    socket_user[request.sid].handle_data(data)