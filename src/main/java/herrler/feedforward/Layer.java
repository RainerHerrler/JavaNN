package herrler.feedforward;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Layer {
	INDArray weights; // Matrix of outputsize x inputsize
	INDArray biases; // Vector of outputsize

	public Layer(INDArray weights, INDArray biases) {
		this.weights = weights;
		this.biases = biases;
	}

	public INDArray forward(INDArray x) {
		INDArray baseValue = weights.mmul(x).add(biases);
		return sigmoid(baseValue);
	}
}


