var e; // epochs

// After page is loaded
$(document).ready(function(){
        // $('.sidenav').sidenav();
        $('.tooltipped').tooltip();
        $('.modal').modal();
        $('.materialboxed').materialbox();
        $('.collapsible').collapsible();

        $("input[name='group']").change(function() {
            $(".manual").toggleClass('disabled_div');
    });

    // socket = io();
    // socket = io.connect('http://localhost:5000/')
    socket = io.connect('http://' + document.domain + ':' + location.port);
    // socket = io.connect('http://127.0.0.1:5000/', {transport: ['websocket'], upgrade: false});

    socket.on('connect', function(){
        console.log('Connected!');
    });

    socket.on('disconnect', function(){
        console.log('Disconnected!');
        $('.loader').addClass('hide');
        $('.hub').addClass('hide');
        $('.tf').addClass('hide');
        alert('Server Disconnected!');
    });

    // for displaying output
    socket.on('hub', function(imgdata, callback){
        $('.loader').addClass('hide');
        $('.hub').addClass('hide');

        download(imgdata);
        display('oimg', imgdata);

        callback();
    });

    // default for send, event is message
    socket.on('message', function(data, callback){
        progress = String(Math.round((parseInt(data['epoch']) / e) * 100)) + '%';
        $('.tf').css('width', progress);
        $('.percentage').text(progress);
        
        if (data['epoch'] == e) {
            $('.loader').addClass('hide');
            $('.percentage').addClass('hide');
            $('.tf').addClass('hide');
        }

        if ('output' in data) {
            download(data['output']);
            display('oimg', data['output']);
        }
        callback();
    });
});

// // for image dimensions
// tmp_img = new Image();
// tmp_img.onload = () => {
//     console.log(tmp_img.name)
//     window[tmp_img.name] = {
//         'width': tmp_img.width,
//         'height': tmp_img.height
//     }
// };

// download button
var download = function(imgdata){
    $('.pulse').removeClass('hide');
    $('.pulse').attr('href', imgdata);
    $('.pulse').attr('download', 'image.jpeg');
};

// for displaying image
var display = function(pos, data){
    $('#' + pos).attr('src', data);
    $('#' + pos).css('display', 'inline-block');
    // $('#' + pos).materialbox();
    $('.' + pos).css('height', '150px');
};

// for loading image
var loadfile = function(event, pos){
    display(pos, URL.createObjectURL(event.target.files[0]));

    // // extracting dimensions
    // tmp_img.name = pos
    // tmp_img.src = URL.createObjectURL(event.target.files[0])
};

var properEncode = function(data){
    // removing data:image/jpeg;base64,
    data = data.toString().replace(/^data:(.*,)?/, '');

    // padding
    len = data.length
    if ((len % 4) > 0){
        data += '='.repeat(4 - (len % 4));
    }
    return data
}

// for transfering images
var transfer = function(){
    var content_image = $('#contentimage')[0].files[0];
    var style_image = $('#styleimage')[0].files[0];

    if ((content_image == undefined) || (style_image == undefined)){
        alert('Please select both content & style images!');
        return;
    }

    if (content_image.type=='image/jpeg' || content_image.type=='image/png' || style_image.type=='image/jpeg' || style_image.type=='image/png'){
        var contentFileReader = new FileReader();
        contentFileReader.readAsDataURL(content_image)              // DataURL (Base64)

        contentFileReader.onload = () => {
            encoded_content = properEncode(contentFileReader.result)                    

            var styleFileReader = new FileReader();
            styleFileReader.readAsDataURL(style_image)

            styleFileReader.onload = () => {
                encoded_style = properEncode(styleFileReader.result)
                // var d = new Date();

                data = {
                    // 'id': d.getTime(),
                    'image1': {
                        // 'name': content_image.name, 
                        'type': content_image.type, 
                        'size': content_image.size,
                        'binary': encoded_content,
                        // 'width': cimg.width,
                        // 'height': cimg.height
                    },
                    'image2': {
                        // 'name': style_image.name, 
                        'type': style_image.type, 
                        'size': style_image.size, 
                        'binary': encoded_style,
                        // 'width': simg.width,
                        // 'height': simg.height
                    }
                }


                if ($("input:radio[name='group']:checked").attr('id') == 'tf') {
                    $('.tf').css('width', '0%');
                    $('.percentage').text('0%');

                    $('.loader').removeClass('hide');
                    $('.percentage').removeClass('hide');
                    $('.tf').removeClass('hide');

                    e = parseInt($('#epochs').val(), 10);

                    data['cfg'] = {
                        'epochs': $('#epochs').val(),
                        'lr': $('#lr').val(),
                        'cwt': $('#cwt').val(),
                        'swt': $('#swt').val()
                    };
                } else {
                    $('.percentage').addClass('hide');
                    $('.loader').removeClass('hide');
                    $('.hub').removeClass('hide');
                }

                socket.emit('data', data);
            };

            styleFileReader.onerror = () => {
                alert('An error occured!\nPlease refresh the page!');
            };
        };

        contentFileReader.onerror = () => {
            alert('An error occured!\nPlease refresh the page!');
        };
    } else {
        alert('Please select a jpeg image!');
    }                
};