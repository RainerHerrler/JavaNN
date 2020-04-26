
package herrler.purejava;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuronLayer {
	
	List<Neuron> neurons;

	public NeuronLayer(List<Neuron> neurons) {
		this.neurons = neurons;
	}
	
	public NeuronLayer(Neuron... neurons) {
		this(Arrays.asList(neurons));
	}
	
	public List<Double> forward(List<Double> inputValues) {
		List<Double> result = new ArrayList<Double>();
		for(Neuron neuron: neurons)
			result.add(neuron.forward(inputValues));
		return result;
	}
}
