# Importing components
from nst import app, socketio, utils, hub_model
from flask import render_template, request

# routings
@app.route("/", methods=['GET'])
def home():
    return render_template(template_name_or_list='index.html')

@socketio.on('connect')
def connected():
    print('Connected!')
    
@socketio.on('disconnect')
def disconnect():
    print('Disconnected!')

def acknowledge():
    print('Output successfully send!')

@socketio.on('data')
def handle_data(data):
    content = hub_model.preprocess(img=utils.base_to_img(data=data['image1']))
    style = hub_model.preprocess(img=utils.base_to_img(data=data['image2']))
    output_img = hub_model.stylize(content_img_tensor=content, style_img_tensor=style)
    output = utils.img_to_base(img=output_img)
    output = u'data:image/jpeg;base64,' + str(output, 'utf-8')
    socketio.emit(event='output', data=str(output), json=False, namespace=None, room=None, callback=acknowledge, include_self=True)