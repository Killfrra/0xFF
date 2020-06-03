var image = document.getElementById('image')
var pickPage = document.getElementById('pick-page')
var cropPage = document.getElementById('crop-page')
var instructions = document.getElementsByClassName('instructions')
var submitButton = document.getElementById('recognize')
var upload = document.getElementById('upload')
var cropper
upload.onchange = function(e){
var url = URL.createObjectURL(upload.files[0])
if(!cropper)
    image.src = url
else
    cropper.replace(url)
}

image.onload = function(e){
    cropper = new Cropper(image, {
        viewMode: 1
    })
    pickPage.style.display = 'none'
    cropPage.style.display = 'block'
    instructions[0].style.display = 'none'
    instructions[1].style.display = 'block'
}
var xhr = new XMLHttpRequest()
submitButton.onclick = function(){
    cropper.disable()
    var data = new FormData()
    var file = upload.files[0]
    data.append('image', file, file.name)
    var box = cropper.getData(true)
    for(var key of ['x', 'y', 'width', 'height'])
        data.append(key, box[key].toString())

    xhr.onload = function(e){
                
        var table = instructions[3].getElementsByTagName('table')[0]
        var results = []
        try {
            results = JSON.parse(xhr.responseText)
        } catch(e) {
            console.error(e)
            return
        }
        var tableHTML = '<tr><th>схожесть</th><th>название</th><th>ссылка</th></tr>'
        for(var r of results){
        var [prob, font, link] = r
        tableHTML += `<tr><td>${ (prob * 100).toFixed(2) }%</td><td>${ font }</td><td><a href="${ link }">Скачать</a></td></tr>`
        }
        table.innerHTML = tableHTML

        instructions[2].style.display = 'none'
        instructions[3].style.display = 'block'
    }
    xhr.onerror = function(e){
        instructions[2].children[0].innerText = 'Была ошибка, вот о чём молчит наука...'
    }
    xhr.open('POST', '/upload')
    xhr.send(data)
    instructions[1].style.display = 'none'
    instructions[2].style.display = 'block'
}