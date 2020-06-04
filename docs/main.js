//jshint asi:true

var requestAttempt = 0

function pseudoXMLHttpRequest(){
    
}
pseudoXMLHttpRequest.prototype.open = function(method, location){
    this.location = location
}
pseudoXMLHttpRequest.prototype.send = function(data){
    if(requestAttempt == 1 && this.onerror){
        this.onerror()
        requestAttempt++
    } else if(this.location == '/upload'){
        if(requestAttempt == 2)
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
        requestAttempt++
    }
}

function empty_func() { }
var fsm = {
    current: { onstart: empty_func, onend: empty_func },
    switch: function (state) {
        this.current.onend()
        error_container.style.display = 'none' //TODO: поместить, куда следует
        state.onstart()
        this.current = state
        this.onafterswitch()
    },
    onafterswitch: empty_func
}

var buttons = document.getElementById('buttons').children
var usedButtons = 0;
function button(innerText, disabled) {
    disabled = disabled || false
    var button = buttons[usedButtons++]
    button.innerText = innerText
    button.disabled = disabled
    button.style.display = 'block'
    return button
}
function disable_unused_buttons() {
    for (var i = usedButtons; i < buttons.length; i++)
        buttons[i].style.display = 'none'
    usedButtons = 0
}

var steps = document.getElementsByClassName('step')

for (var s of steps)
    s.disabled = true

var current_step = 0
function step(num) {
    steps[current_step].classList.remove('current')
    steps[current_step].disabled = false
    steps[num].classList.add('current')
    steps[num].disabled = true
    current_step = num
}

var image = document.getElementById('image-to-crop')
var cropper_inited = false
var cropper = new Cropper(image, {
    viewMode: 1,
    guides: false,
    center: false,
    background: false,
    //minCropBoxWidth: 64,
    //minCropBoxHeight: 64
})

var upload_field = document.getElementById('upload-field')
function update_cropper() {
    var url = URL.createObjectURL(upload_field.files[0])
    cropper.replace(url)
    cropper_inited = true
}
upload_field.onchange = function () {
    update_cropper()
    fsm.switch(states.crop_screen)
}

var cropper_container = document.getElementById('cropper-container')
var instructions = document.getElementById('instructions')
var results_container = document.getElementById('results-container')
var results = results_container.getElementsByClassName('result')
var spinner = document.getElementById('spinner')
var error_container = document.getElementById('error-container')
var comment_container = document.getElementById('comment')
var comment_send_btn = comment_container.getElementsByTagName('button')[0]
var comment_textarea = comment_container.getElementsByTagName('textarea')[0]
var notification = document.getElementById('notification')
var notification_close_btn = notification.getElementsByTagName('span')[0]
notification_close_btn.onclick = function(){
    notification.style.display = 'none'
}

function disable_all() {
    cropper_container.style.display = 'none'
    instructions.style.display = 'none'
    results_container.style.display = 'none'
    comment_container.style.display = 'none'
    error_container.style.display = 'none'
}

comment_send_btn.onclick = function () {
    if (!error_text && !comment_textarea)
        return;

    var data = {
        'comment': comment_textarea.value,
        'error': error_text
    }
    var xhr = new pseudoXMLHttpRequest()
    xhr.open('POST', '/comment')
    xhr.send(data)
    error_container.style.display = 'none'
    comment_textarea.placeholder = 'Спасибо за ваш отзыв!'
    comment_textarea.value = ''
    //error_text = ''
    comment_send_btn.innerText = 'Отправить ещё'
}

var states = {
    start_screen: {
        onstart: function () {
            step(0)
            var can_continue = upload_field.files.length > 0
            var btn = button('Далее →', !can_continue)
            steps[1].disabled = !can_continue
            btn.onclick = upload_field.onchange
        },
        onend: empty_func
    },
    crop_screen: {
        onstart: function () {
            step(1)
            if (!cropper_inited)
                update_cropper()
            cropper.enable()
            cropper_container.style.display = 'block'
            instructions.style.display = 'block'
            var prev_btn = button('← Назад')
            prev_btn.onclick = () => fsm.switch(states.start_screen)
            var next_btn = button('Распознать →')
            next_btn.onclick = () => fsm.switch(states.loading_screen)
            steps[2].disabled = false
        },
        onend: function () {
            cropper_container.style.display = 'none'
            instructions.style.display = 'none'
        }
    },
    loading_screen: {
        onstart: function () {
            step(2)
            cropper.disable()
            cropper_container.style.display = 'block'
            results_container.style.display = 'block'
            spinner.style.display = 'block'
            var btn = button('Отменить')
            btn.onclick = () => fsm.switch(states.crop_screen)

            // upload block

            var data = new FormData()
            var file = upload_field.files[0]
            data.append('image', file, file.name)
            var box = cropper.getData(true)
            for (var key of ['x', 'y', 'width', 'height'])
                data.append(key, box[key].toString())

            var xhr = new pseudoXMLHttpRequest()
            xhr.onload = function (e) {
                // пришли результаты, но до этого пользователь нажал "отменить"
                if(fsm.current !== states.loading_screen)
                    return // результат же ему больше не нужен, правда?

                var response = []
                try {
                    response = JSON.parse(xhr.responseText)
                    if(response.error)
                        throw response.error
                } catch (err) {
                    results_container.style.display = 'none'
                    show_error(err)
                    return
                }
                for (var i = 0; i < results.length && i < response.length; i++) {
                    var img = results[i].children[0]
                    var [, font, link] = response[i]
                    img.src = 'previews/' + font + '.jpg'
                    img.alt = font
                    results[i].href = link
                }

                spinner.style.display = 'none'
                comment_textarea.placeholder = 'Ну как? Мы угадали? :)\nНапишите отзыв о нашей работе'
                comment_container.style.display = 'block'
            }
            xhr.onerror = function () {
                results_container.style.display = 'none'
                show_error('Произошла ошибка при отправке данных на сервер')
            }
            xhr.open('POST', '/upload')
            xhr.send(data)
        },
        onend: function () {
            cropper_container.style.display = 'none'
            results_container.style.display = 'none'
            comment_container.style.display = 'none'
        }
    }
}

var error_text = ''
function show_error(e) {
    error_text = e.toString()
    error_container.children[1].innerText = error_text
    error_container.style.display = 'block'
    comment_send_btn.innerText = 'Отправить'
    comment_textarea.placeholder = 'Будем рады любому отклику'
    comment_container.style.display = 'block'
}
window.onerror = function (event, source, lineno, colno, error) {
    error_text = source + ':' + lineno + ':' + colno + ': ' + event
    if (error && error.stack)
        error_text += '\n' + error.stack
    show_error(error_text)
}

disable_all()

steps[0].onclick = () => fsm.switch(states.start_screen)
steps[1].onclick = () => fsm.switch(states.crop_screen)
steps[2].onclick = () => fsm.switch(states.loading_screen)

fsm.onafterswitch = disable_unused_buttons
fsm.switch(states.start_screen)