function getScreenSize() {
    var dpi = document.createElement("div");   
    dpi.id = "dpi"
    dpi.style.height = "1in";
    dpi.style.width = "1in";
    dpi.style.left = "100%";
    dpi.style.position =  "fixed";
    dpi.style.top = "100%";   

    document.body.appendChild(dpi); 

    var dpi_x = document.getElementById('dpi').offsetWidth;
    var dpi_y = document.getElementById('dpi').offsetHeight;

    var width = screen.width / dpi_x;
    var height = screen.height / dpi_y;

    var screenSize = Math.sqrt(width * width +  height * height);

    document.body.removeChild(dpi); 

    return screenSize

}

function setScreenZoom(screenSize, body, zoom_13, zoom_15){

    if(screenSize > 20  && screenSize < 25){
        body.style.zoom = zoom_13;
        body.style.fontSize = "1.75rem";
    }else if (screenSize >= 25 && screenSize < 35){
        body.style.zoom = zoom_15;
    }

}

const screenSize = getScreenSize();
var body = document.getElementsByTagName('body')[0];


setScreenZoom(screenSize, body, 0.90, 0.95);
