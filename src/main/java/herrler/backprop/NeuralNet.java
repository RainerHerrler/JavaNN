package herrler.backprop;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuralNet {

	final Layer[] layers;
	float learningRate = 0.5f;

	public NeuralNet(Layer... layers) {
		this.layers = layers;
	}

	public INDArray forward(INDArray a) {
		for (Layer layer : layers) {
			a = layer.forward(a);
		}
		return a;
	}
	
	INDArray forward(INDArray a, int level) {
		int count=0;
		for (Layer layer : layers) {
			a = layer.forward(a);
			if (count++>=level)
				return a;
		}
		return a;		
	}
	
	public int getOutputSize() {
		return (int) layers[layers.length - 1].getOutputSize();
	}

	public int getInputSize() {
		return (int) layers[0].getInputSize();
	}

	public float fit(INDArray x, INDArray yGiven) {
		INDArray y = forward(x);
		float e1 = calcMSEError(yGiven, y);
		INDArray localGradients = calcMSEGradients(yGiven, y);
		for (int i = layers.length - 1; i >= 0; i--) {
			localGradients = layers[i].backprop(localGradients, learningRate);
		}
		System.out.printf("Error : %f  \n", e1);
		return e1;
	}
	


	private static INDArray calcMSEGradients(INDArray yGround, INDArray y) {
		return yGround.sub(y).mul(-1);
	}

	private static float calcMSEError(INDArray yGround, INDArray y) {
		INDArray diff = yGround.sub(y);
		diff = diff.muli(diff);
		INDArray sum = Nd4j.sum(diff);
		return sum.getFloat(0, 0);
	}

	public static void main(String[] args) {
		Nd4j.getRandom().setSeed(7000);
		Layer layer1 = new Layer(2, 2, ActivationFunction.LeakyRelu);
		Layer layer2 = new Layer(2, 1, ActivationFunction.Sigmoid);
		NeuralNet net = new NeuralNet(layer1, layer2);
		
		ScatterPlot plot = ScatterPlot.openWindow();
		ErrorPlot errorPlot = ErrorPlot.openWindow();

		for (int s = 0; s < 100001; s++) {
			INDArray x = Nd4j.rand(2, 1); // create random 2d verctor as a sample
			
			float expected = (x.getFloat(0) > 0.5f && x.getFloat(1) > 0.5f) ? 1f : 0f;
			INDArray labelvector = Nd4j.create(new float[] { expected }, new int[] { 1, 1 }, 'c');
			float error = net.fit(x,labelvector);
			errorPlot.add(error);
			if (s!=0 && s % 2000 ==0) {
				plot.showOutputs("Samples "+s, net);
			}
		}
		net.printParameters(); // print resulting net
	}

	/**
	 * Print parameters of all Layers
	 */
	public void printParameters() {
		int i = 0;
		for (Layer layer : layers) {
			i++;
			System.out.println("Layer "+i);
			System.out.println("Weights:" + layer.weights);
			System.out.println("Biases:" + layer.biases.transpose());
		}
	}
}
