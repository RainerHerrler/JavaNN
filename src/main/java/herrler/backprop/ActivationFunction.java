package herrler.backprop;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public abstract class ActivationFunction {
	
	public static ActivationFunction LeakyRelu = new LeakyReluFunction();
	public static ActivationFunction Sigmoid = new SigmoidFunction();

	public static class LeakyReluFunction extends ActivationFunction {
		@Override
		INDArray activate(INDArray basisValues) {
			return Transforms.leakyRelu(basisValues);
		}

		@Override
		INDArray derivation(INDArray baseValues) {
			return Transforms.leakyReluDerivative(baseValues, 0);
		}
	}

	public static class SigmoidFunction extends ActivationFunction {
		@Override
		INDArray activate(INDArray basisValues) {
			return Transforms.sigmoid(basisValues);
		}

		@Override
		INDArray derivation(INDArray baseValues) {
			return Transforms.sigmoidDerivative(baseValues);
		}
	}

	abstract INDArray activate(INDArray basisValues);

	abstract INDArray derivation(INDArray baseValues);
}
