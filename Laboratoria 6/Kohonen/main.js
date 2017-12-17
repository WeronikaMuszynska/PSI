var synaptic = require('synaptic');
var fs = require('fs');
var Layer = synaptic.Layer,
    Network = synaptic.Network;

var inputLayer = new Layer(25);
var hiddenLayer = new Layer(5);
var outputLayer = new Layer(20);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var network = new Network({
    input: inputLayer,
    hidden: [hiddenLayer],
	output: outputLayer
});


var som = require('node-som');
var somInstance = new som({
	inputLength: 25,
	maxClusters: 10,
	loggingEnabled:false,
	target: 0.1,
	scale:{
		min:0,
		max:10
	}
});
somInstance.train(trainingData);

var sample = [0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1];
var group = somInstance.classify(sample);
console.log(getFlowerName2(group));






var trainingData = [];

fs.readFile('iris.txt','utf8',function(err, data){
	if(err) throw err;

	var lines = data.split("\n");
	
	for(var i = 0; i < lines.length; i++){
		var line = lines[i].trim();
		var splitLine = line.split(",");
		var input = splitLine.slice(0, 25);

		var output = splitLine[25]=='A' ? [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
					: splitLine[25] == 'B' ? [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'C' ? [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'D' ? [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'E' ? [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'F' ? [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'G' ? [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'H' ? [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'I' ? [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'J' ? [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'K' ? [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'L' ? [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
					: splitLine[25] == 'M' ? [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
					: splitLine[25] == 'N' ? [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
					: splitLine[25] == 'O' ? [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
					: splitLine[25] == 'P' ? [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
					: splitLine[25] == 'R' ? [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
					: splitLine[25] == 'S' ? [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
					: splitLine[25] == 'T' ? [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
					: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1];

		trainingData.push({
			input: input,
			output: output
		});
	}

	var learningRate = .01;
	for (var i = 1; i <= 10000; i++)
	{
		for(var j = 0; j < trainingData.length; j++){
			network.activate(trainingData[j].input);
			network.propagate(learningRate, trainingData[j].output);
		}
		
	}

	
	fs.readFile('testing_data.txt','utf8',function(err, data){
		if(err) throw err;
		console.log("\n\nResults\n===============================\n");
		var lines = data.split("\n");
		for(var i = 0; i < lines.length; i++){
			var input = lines[i].trim().split(",");
			var result = getFlowerName(network.activate(input));
			console.log(lines[i].trim()+" => "+result);
		}
	});
});


function getLargestIndex(arr){
	var result = 0;
	for(var i = 1; i < arr.length; i++)
		if(arr[i] > arr[result])
			result = i;
	return result;
}

function getFlowerName(arr){
	var index = getLargestIndex(arr);
	if(index == 0) return "A";
	if(index == 1) return "B";
	if(index == 2) return "C";
	if(index == 3) return "D";
	if(index == 4) return "E";
	if(index == 5) return "F";
	if(index == 6) return "G";
	if(index == 7) return "H";
	if(index == 8) return "I";
	if(index == 9) return "J";
	if(index == 10) return "K";
	if(index == 11) return "L";
	if(index == 12) return "M";
	if(index == 13) return "N";
	if(index == 14) return "O";
	if(index == 15) return "P";
	if(index == 16) return "R";
	if(index == 17) return "S";
	if(index == 18) return "T";
	return "U";
}
function getFlowerName2(index){
	if(index == 0) return "A";
	if(index == 1) return "B";
	if(index == 2) return "C";
	if(index == 3) return "D";
	if(index == 4) return "E";
	if(index == 5) return "F";
	if(index == 6) return "G";
	if(index == 7) return "H";
	if(index == 8) return "I";
	if(index == 9) return "J";
	if(index == 10) return "K";
	if(index == 11) return "L";
	if(index == 12) return "M";
	if(index == 13) return "N";
	if(index == 14) return "O";
	if(index == 15) return "P";
	if(index == 16) return "R";
	if(index == 17) return "S";
	if(index == 18) return "T";
	return "U";
}




