//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var heatDisplay = document.getElementById("heatmap-display");
var predLinks = document.getElementById("predLinks");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var fileSelect = document.getElementById("file-upload");
var imageRow = document.getElementById('image-row');


//========================================================================
// Main button events
//========================================================================

fileSelect.onchange = evt => {
  const [file] = fileSelect.files
  if (file) {
    previewFile(file);
  }
}

function submitImage() {
  // action for the submit button
  console.log("submit");
  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

  show(predLinks);
  show(loader);

  // call the predict function of the backend
  predictImage(imageDisplay.src);
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  // imagePreview.src = "";
  imageDisplay.src = "";
  heatDisplay.src = "";
  predResult.innerHTML = "";

  // hide(imagePreview);
  hide(imageDisplay);
  hide(heatDisplay);
  hide(loader);
  hide(predResult);
  hide(predLinks);
  remove(imageRow, "gx-5");
  remove(imageRow, "row");
}

function previewFile(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    // reset
    hide(predLinks);
    hide(heatDisplay);
    remove(imageRow, "gx-5");
    remove(imageRow, "row");
    predResult.innerHTML = "";
    heatDisplay.src = "";

    displayImage(reader.result, "image-display");
  };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  hide(loader);
  add(imageRow, "gx-5");
  add(imageRow, "row");

  if (data.result != "") {
    predResult.innerHTML = "Prediction: "+ data.result + " (" + data.probability + ")";
  }
  heatDisplay.src = data.heatmap;
  show(heatDisplay);
  show(predResult);
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function add(el, tag) {
  // show an element
  el.classList.add(tag);
}

function remove(el, tag) {
  // show an element
  el.classList.remove(tag);
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}