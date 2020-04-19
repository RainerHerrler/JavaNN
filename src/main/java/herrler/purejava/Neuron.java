package herrler.purejava;

import java.util.List;

public class Neuron {

	List<Double> weights;
	double bias;
	
	public Neuron(List<Double> weights, double bias) {
		this.weights = weights;
		this.bias = bias;
	}

	public double forward(List<Double> inputValues) {
		double sum = bias;
		for (int i = 0; i < inputValues.size(); i++) {
			sum += inputValues.get(i) * weights.get(i);
		}
		return sigmoid(sum);
	}

	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
}
