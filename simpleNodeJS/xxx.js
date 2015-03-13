var formidable = require('formidable'),
    http = require('http'),
    util = require('util'),
    fs_ext   = require('fs-extra'),
    path = require('path'),
    fs = require('fs');


 
http.createServer(function(req, res) {
  /* Process the form uploads */
  if (req.url == '/upload' && req.method.toLowerCase() == 'post') {
    var dst = '';
    var form = new formidable.IncomingForm();
    //form.uploadDir = "./upload/";
    form.parse(req, function(err, fields, files) {

    });
    form.on('progress', function(bytesReceived, bytesExpected) {
        var percent_complete = (bytesReceived / bytesExpected) * 100;
        console.log(percent_complete.toFixed(2));
    });
 
    form.on('error', function(err) {
        console.error(err);
    });

    form.on('end', function(fields, files) {
      var arg = ["./upload/"];
      var new_location = './upload/';
       for (var i = 0; i < this.openedFiles.length; i++) {    
          var temp_path = this.openedFiles[i].path;
        /* The file name of the uploaded file */
        var file_name = this.openedFiles[i].name;
        /* Location where we want to copy the uploaded file */
	var tmp = new_location+file_name;
	if (path.extname(file_name).toLowerCase() == '.jpg') arg.push(tmp);
 
        fs_ext.copy(temp_path, tmp, function(err) {  
            if (err) {
                console.error(err);
            } else {
                console.log("success!");
            }
        });
      }


        /* Temporary location of our uploaded file */
      
  // output here  
        
	 var spawn = require('child_process').spawn;
	  var child = spawn('./ImageStitching', arg);
	  console.log(arg);
	  child.stdout.on('data', function(chunk) {
	    dst = chunk.toString();
	    console.log('stdout: ' + dst);
	    fs_ext.readFile(dst, function(err, data) {
	  if (err) throw err; 
	  res.writeHead(200, {'content-type': 'image/jpeg'});
	  res.end(data);
	  });
	});
  });

 
    return;
  }
 
  /* Display the file upload form. */
  res.writeHead(200, {'content-type': 'text/html'});
  res.end(
    '<form action="/upload" enctype="multipart/form-data" method="post">'+
    '<input type="text" name="title"><br>'+
    '<input type="file" name="upload" multiple="multiple"><br>'+
    '<input type="submit" value="Upload">'+
    '</form>'
  );
 
 
}).listen(8080); 


