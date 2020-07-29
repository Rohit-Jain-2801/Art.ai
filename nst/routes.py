# Importing components
from nst import app, socketio, utils
from nst.models import hub_model, tf_model
from flask import render_template, url_for

# routings
@app.route("/", methods=['GET'])
def home():
    return render_template(template_name_or_list='index.html')

connection = None

@socketio.on('connect')
def connected():
    global connection
    connection = True
    print('Connected!')


@socketio.on('disconnect')
def disconnect():
    global connection
    connection = False
    print('Disconnected!')


def acknowledge():
    print('Data sent!')


@socketio.on('data')
def handle_data(data):
    content_img = utils.base_to_img(data=data['image1'])
    style_img = utils.base_to_img(data=data['image2'])
    
    if 'cfg' in data.keys():
        for epoch, update, output_img in tf_model.run_style_transfer(
                content_img=content_img, 
                style_img=style_img,
                num_iterations=int(data['cfg']['epochs']),
                content_weight=float(data['cfg']['cwt']),
                style_weight=float(data['cfg']['swt']),
                learning_rate=float(data['cfg']['lr'])
            ):
            
            if update:
                output = utils.img_to_baseuri(img=output_img)
                out = {
                    'epoch': epoch,
                    'output': output
                }
            else:
                out = {
                    'epoch': epoch,
                }
            
            if connection:
                socketio.send(data=out, json=False, namespace=None, room=None, callback=acknowledge, include_self=True)
                socketio.sleep(1)
            else:
                break

    else:
        output_img = hub_model.run_style_transfer(content_img=content_img, style_img=style_img)
        output = utils.img_to_baseuri(img=output_img)
        socketio.emit(event='hub', data=str(output), json=False, namespace=None, room=None, callback=acknowledge, include_self=True)