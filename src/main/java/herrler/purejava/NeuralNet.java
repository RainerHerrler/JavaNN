package herrler.purejava;

import java.util.List;

public class NeuralNet {

	final List<NeuronLayer> layers;

	public NeuralNet(List<NeuronLayer> layers) {
		this.layers = layers;
	}

	public List<Double> forward(List<Double> a) {
		for (NeuronLayer layer : layers) {
			a = layer.forward(a);
		}
		return a;
	}
}


//public int getOutputSize() {
//return (int) layers[layers.length - 1].neurons.size();
//}
//
//public int getInputSize() {
//return (int) layers[0].neurons.get(0).weights.size();
//}