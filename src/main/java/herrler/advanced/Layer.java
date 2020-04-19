package herrler.advanced;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoidDerivative;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Layer {
	INDArray weights; // Matrix of outputsize x inputsize
	INDArray biases; // Vector of outputsize

	private INDArray baseValue; // cached baseValue
	private INDArray activations; // cached activation
	private INDArray input;

	public Layer(INDArray weights, INDArray biases) {
		this.weights = weights;
		this.biases = biases;
	}

	public Layer(int inputSize, int outputSize) {
		this(Nd4j.rand(outputSize, inputSize), Nd4j.rand(outputSize, 1));
	}

	public INDArray forward(INDArray x) {
		input = x;
		baseValue = weights.mmul(x).add(biases);
		activations = sigmoid(baseValue);
		return activations;
	}

	public INDArray backprop(INDArray upStreamGradients, double learningRate) {
		INDArray gradientsWithRespectToZ = sigmoidDerivative(baseValue).mul(upStreamGradients);
		INDArray gradientsWithRepectToW = gradientsWithRespectToZ.mmul(input.transpose());
		INDArray downStreamGradient = Nd4j.sum(gradientsWithRepectToW, 0).reshape(2, 1);

		weights = weights.addi(gradientsWithRepectToW.mul(-learningRate));
		biases = biases.addi(gradientsWithRespectToZ.mul(-learningRate));
		return downStreamGradient;
	}
	
	void printParameters() {
		System.out.println("Weights:" + weights);
		System.out.println("Biases:" + biases.transpose());
	}

}
