
function doLoad() {
  	this.video = document.getElementById('video');
  	this.canvas = document.getElementById('canvas');
  	this.cube = null;
  	this.data = null;

  	initWebGL();

  	d3.json("frame_data.json",function(error,data){
        if (error) return console.warn(error);
        this.data = data;
    });
}

function initWebGL() {
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera( 45, this.canvas.width / this.canvas.height, 0.1, 1000 );

    var canvas = document.getElementById('canvas');
    var renderer = new THREE.WebGLRenderer({canvas: canvas});
    renderer.setSize( canvas.width, canvas.height );

    var geometry = new THREE.BoxGeometry( 1, 1, 1 );
    var material = new THREE.MeshNormalMaterial();
    var cube = new THREE.Mesh( geometry, material );
    scene.add( cube );
    this.cube = cube;

    camera.position.z = 5;

    var render = function () {
        requestAnimationFrame( render );

        //cube.rotation.x += 0.01;
        //cube.rotation.y += 0.01;
        //cube.rotation.z += 0.01;
        var frame_idx = getFrameIdx();
        if (!this.video.paused && !this.video.ended) {
        	console.log(frame_idx);
		    renderInCanvas(frame_idx);
		}


        renderer.render(scene, camera);
    };

    render();
}

function renderInCanvas(frame_idx) {
	if (this.data != null){	
		var frames = this.data.filter(frame => frame['frame_idx'] == frame_idx);
		if (frames.length == 0) {

		} else {
			var face = frames[0]["faces"][0];
			if (face != null) {
				var pose = face["faceAttributes"]["headPose"];
				console.log(pose);
				this.cube.rotation.x = THREE.Math.degToRad(pose["pitch"]);
				this.cube.rotation.y = THREE.Math.degToRad(pose["yaw"]);
				this.cube.rotation.z = -THREE.Math.degToRad(pose["roll"]);
			}

		}

	}
}

function getFrameIdx() {
  var video = document.getElementsByTagName('video')[0];
  return Math.round(video.currentTime * 30.0);
}