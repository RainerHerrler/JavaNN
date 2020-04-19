package herrler.backprop;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Layer {
	INDArray weights; // Matrix of outputsize x inputsize
	INDArray biases; // Vector of outputsize
	ActivationFunction activation = new ActivationFunction.SigmoidFunction();

	private INDArray baseValue; // cached baseValue
	private INDArray activations; // cached activation
	private INDArray input;

	public Layer(INDArray weights, INDArray biases) {
		this.weights = weights;
		this.biases = biases;
	}
		
	public Layer(int inputSize, int outputSize, ActivationFunction activation) {
		this(Nd4j.rand(outputSize, inputSize).mul(3), Nd4j.rand(outputSize, 1).mul(3));
		this.activation = activation;
	}

	public INDArray forward(INDArray x) {
		input = x;
		baseValue = weights.mmul(x).add(biases);
		activations = activation.activate(baseValue);
		return activations;
	}
		
	
	public INDArray backprop(INDArray upStreamGradients, double learningRate) {
		INDArray gradientsWithRespectToZ = activation.derivation(baseValue).mul(upStreamGradients);
		INDArray gradientsWithRepectToW = gradientsWithRespectToZ.mmul(input.transpose()).div(input.size(0));
		INDArray gradientsWithRepectTob = Nd4j.sum(gradientsWithRespectToZ, 0).div(input.size(0));

		INDArray downStreamGradient = weights.transpose().mmul(gradientsWithRespectToZ);
		weights = weights.addi(gradientsWithRepectToW.mul(-learningRate));
		biases = biases.addi(gradientsWithRepectTob.mul(-learningRate));
		return downStreamGradient;
	}
	
	void printParameters() {
		System.out.println("Weights:" + weights);
		System.out.println("Biases:" + biases.transpose());
	}

	public int getInputSize() {
		return (int) weights.size(1);
	}
	
	public int getOutputSize() {
		return (int) weights.size(0);
	}
}
