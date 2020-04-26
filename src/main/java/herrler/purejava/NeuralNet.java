package herrler.purejava;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.nd4j.nativeblas.Nd4jCpu.neq_scalar;

import io.netty.util.collection.CharCollections;

public class NeuralNet {

	final List<NeuronLayer> layers;

	public NeuralNet(List<NeuronLayer> layers) {
		this.layers = layers;
	}
	
	public NeuralNet(NeuronLayer... layers) {
		this.layers = Arrays.asList(layers);
	}
	
	public List<Double> forward(List<Double> a) {
		for (NeuronLayer layer : layers) {
			a = layer.forward(a);
		}
		return a;
	}
	
	public static void main(String[] args) {
		Neuron a1 = new Neuron(createList(1000.0,0.0), -500);
		Neuron a2 = new Neuron(createList(0.0,1000.0), -500);
		Neuron y1 = new Neuron(createList(10.0,10.0), -18);
		
		NeuronLayer hiddenLayer = new NeuronLayer(a1,a2);
		NeuronLayer outputLayer = new NeuronLayer(y1);
		
		NeuralNet net = new NeuralNet(hiddenLayer, outputLayer);
		
		List<Double> output = net.forward(createList(0.4, 0.6));
		System.out.printf("%.4f", output.get(0).floatValue());
	}
	
	public static List<Double> createList(Double... values) {
		return Arrays.asList(values);
	}
}




//public int getOutputSize() {
//return (int) layers[layers.length - 1].neurons.size();
//}
//
//public int getInputSize() {
//return (int) layers[0].neurons.get(0).weights.size();
//}