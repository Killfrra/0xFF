//jshint asi:true

/*
function pseudoXMLHttpRequest(){
    this.requestAttempt = 0
}
pseudoXMLHttpRequest.prototype.open = function(method, location){
    this.location = location
}
pseudoXMLHttpRequest.prototype.send = function(data){
    if(this.requestAttempt == 1 && this.onerror){
        this.onerror()
        this.requestAttempt++
    } else if(this.location == '/upload'){
        if(this.requestAttempt == 2)
            this.responseText = JSON.stringify({
                error: 'ошибка пришла с сервера'
            })
        else this.responseText = JSON.stringify(
                [
                    [5, 'Golos Text_Regular', 'about:blank'],
                    [4, 'NotoSerif-Regular', 'about:blank'],
                    [3, 'Oswald-Regular', 'about:blank'],
                    [2, 'Phenomena-Regular', 'about:blank'],
                    [1, 'Sreda-Regular', 'about:blank']
                 ]
            )
        
        setTimeout(this.onload, 500)
        this.requestAttempt++
    }
}
*/
var pseudoXMLHttpRequest = XMLHttpRequest

function css_state(state){
    document.body.setAttribute('state', state)
    document.body.removeAttribute('error')
}

var next_btn = document.getElementById('next-btn')
var prev_btn = document.getElementById('prev-btn')
var steps = document.getElementsByClassName('step')
var upload_container = document.getElementById('upload-container')
var upload_field = document.getElementById('upload-field')
var upload_droparea = document.getElementById('upload-droparea')
var image = document.getElementById('image-to-crop')
var preview_container = document.getElementById('preview-container')
var preview = document.getElementById('preview')
var results_container = document.getElementById('results-container')
var results = results_container.getElementsByTagName('tbody')[0]
var error_container = document.getElementById('error-container')
var error = document.getElementById('error')
var comment_container = document.getElementById('comment-container') // not used
var comment_textarea = comment_container.getElementsByTagName('textarea')[0]
var comment_send_btn = comment_container.getElementsByTagName('button')[0]
var offscreen_canvas = document.getElementById('offscreen-canvas')

var file, cropper, image_url, cropper_inited = false;
function update_cropper(){
    image_url = URL.createObjectURL(file)    
    if(!cropper){
        image.src = image_url
        cropper = new Cropper(image, {
            viewMode: 1,
            preview: preview,
            guides: false,
            center: false,
            background: false,
            autoCropArea: 0.25,
            //minCropBoxWidth: 64,
            //minCropBoxHeight: 64
        })
    } else {
        cropper.enable()
        cropper.replace(image_url)
    }
    cropper_inited = true
}

function preventDefault(e){
    e.preventDefault();
    e.stopPropagation();
}

['dragover', 'dragenter', 'dragleave', 'drop'].forEach(function(e) {
    upload_container.addEventListener(e, preventDefault, false)
})

upload_container.addEventListener('drop', function(e){
    file = e.dataTransfer.files[0]
    update_cropper()
    fsm.next_step()
}, false)

upload_field.onchange = function(){
    file = upload_field.files[0]
    update_cropper()
    fsm.next_step()
}

function empty_func(){}

var fsm = {
    step: 0,
    state: { start: empty_func, end: empty_func },
    next_step: function(){
        var step = Math.min(this.steps.length - 1, this.step + 1)
        var state = this.steps[step]
        this.switch(step, state, state.name)
    },
    prev_step: function(){
        var step = Math.max(0, this.step - 1)
        var state = this.steps[step]
        this.switch(step, state, state.name)
    },
    switch: function(step, state, name){
        this.step = step
        this.state.end()
        this.state = state
        css_state(name)
        state.start()
    }
}

fsm.upload_state = function upload() {
    this.switch(0, this.upload_state, 'upload')
}
fsm.upload_state.start = function () {
    var can_continue = upload_field.files.length > 0
    prev_btn.disabled = true
    next_btn.disabled = !can_continue
    steps[0].disabled = true
    steps[1].disabled = !can_continue
}
fsm.upload_state.end = function(){
    steps[0].disabled = false
    next_btn.disabled = false
    prev_btn.disabled = false
}

fsm.crop_state = function crop() {
    this.switch(1, this.crop_state, 'crop')
}
fsm.crop_state.start = function () {
    if (!cropper_inited){
        file = upload_field.files[0]
        update_cropper()
    } else
        cropper.enable()
    steps[1].disabled = true
    steps[2].disabled = false
}
fsm.crop_state.end = function(){
    cropper.disable()
    steps[1].disabled = false
}

fsm.loading_state = function loading() {
    this.switch(2, this.loading_state, 'loading')
}
function grayscale(ctx, imageData) {
    var data = imageData.data
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg;      // red
        data[i + 1] = avg;  // green
        data[i + 2] = avg;  // blue
    }
    ctx.putImageData(imageData, 0, 0);
}
fsm.loading_state.start = function(){
    /*
    var data = new FormData()
    data.append('image', file, file.name)
    var box = cropper.getData(true)
    ;(['x', 'y', 'width', 'height']).forEach(function(key){ //TODO: rewrite
        data.append(key, box[key].toString())
    })
    */
    var xhr = new pseudoXMLHttpRequest()
    xhr.onload = function (e) {
        // пришли результаты, но до этого пользователь струсил и нажал "назад"
        if(fsm.state !== fsm.loading_state)
            return // результат же ему больше не нужен, правда?

        var response = []
        try {
            response = JSON.parse(xhr.responseText)
            if(response.error){
                show_error(response.error)
                return
            }
        } catch (err) {
            show_error(err)
            return
        }
        while(results.childElementCount > 1)
            results.children[results.childElementCount - 1].remove()
        
        var table_src = ''
        for (var i = 0; i < response.length; i++) {
            var font = response[i][0],
                license = response[i][1],
                link = response[i][2] || '#'
            table_src += '<tr>'
            table_src += '<td><img src="static/previews/' + font + '.jpg" alt="' + font + '"></img></td>'
            table_src += '<td>' + font + '</td>'
            table_src += '<td>' + license + '</td>'
            table_src += '<td><a class="fas" href="' + link + '" target="_blank"></a></td>'
            table_src += '</tr>'
        }
        results.innerHTML += table_src

        fsm.results_state()

    }
    xhr.onerror = function () {
        show_error('Произошла ошибка при отправке данных на сервер')
    }
    var bbox = cropper.getCropBoxData()
    while(bbox.height / 2 > 127){
        bbox.width /= 2
        bbox.height /= 2
    }
    var canvas = cropper.getCroppedCanvas({
        width: bbox.width,
        height: bbox.height,
        //maxHeight: 127,
        //imageSmoothingEnabled: false,
        //imageSmoothingQuality: 'low'
    })
    console.log(canvas.width, canvas.height)
    var ctx = canvas.getContext('2d')
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    grayscale(ctx, imageData)
    canvas.toBlob(function(blob){
        var data = new FormData()
        data.append('image', blob)
        xhr.open('POST', '/upload')
        xhr.send(data)
    }, 'image/jpeg')
    //xhr.open('POST', '/upload')
    //xhr.send(data)
}
fsm.loading_state.end = function(){
    steps[1].disabled = false
}

fsm.results_state = function results() {
    this.switch(2, this.results_state, 'results')
}
fsm.results_state.start = function () {
    //function render_preview(){
        cropper.disabled = false
        preview.cropperPreview.width = preview_container.offsetWidth
        preview.cropperPreview.height = preview_container.offsetHeight
        cropper.preview()
        cropper.disabled = true
    //}
    results_container.scrollIntoView({
        behavior: 'smooth'
    })
    next_btn.disabled = true
    prev_btn.disabled = false
    steps[2].disabled = true
}
fsm.results_state.end = function(){
    steps[2].disabled = false
    next_btn.disabled = false
}

var error_text = ''
function show_error(e, comment){
    if(comment)
        error_text = e
    document.body.setAttribute('error', comment ? 'comment' : 'true')
    error.innerText = e
    error_container.scrollIntoView({
        behavior: 'smooth'
    })
}

window.onerror = function (event, source, lineno, colno, error) {
    error_text = source + ':' + lineno + ':' + colno + ': ' + event
    if (error && error.stack)
        error_text += '\n' + error.stack
    show_error(error_text, true)
}

function submit_comment(){
    var comment = comment_textarea.value
    if (!error_text && !comment)
        return;

    var data = {}
    if(error_text)
        data.error = error_text
    if(comment)
        data.comment = comment
    
    var xhr = new pseudoXMLHttpRequest()
    xhr.open('POST', '/comment')
    xhr.send(JSON.stringify(data))
    
    document.body.removeAttribute('error')
    comment_textarea.placeholder = 'Спасибо за ваш отзыв!'
    comment_textarea.value = ''
    //error_text = ''
    comment_send_btn.innerText = 'Отправить ещё'
}

fsm.steps = [ fsm.upload_state, fsm.crop_state, fsm.loading_state ]

for(var i = 0; i < steps.length; i++)
    steps[i].disabled = true

fsm.upload_state()