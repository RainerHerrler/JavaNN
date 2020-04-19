package herrler.feedforward;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kotlin.Pair;

public class NeuralNet {
	final Layer[] layers;

	public NeuralNet(Layer... layers) {
		this.layers = layers;
	}
	
	public INDArray forward(float... inputs) {
		return forward(Nd4j.create(inputs, new int[] { inputs.length, 1 }, 'c'));
	}

	public INDArray forward(INDArray a) {
		for (Layer layer : layers) {
			a = layer.forward(a);
		}
		return a;
	}

	public int getOutputSize() {
		return (int) layers[layers.length - 1].weights.size(0);
	}

	public int getInputSize() {
		return (int) layers[0].weights.size(0);
	}

	public static void main(String[] args) {
		INDArray weights1 = Nd4j.create(new float[] { 0, 1000, 1000, 0 }, new int[] { 2, 2 }, 'c');
		INDArray biases1 = Nd4j.create(new float[] { -500f, -500f }, new int[] { 2, 1 }, 'c');
		Layer layer1 = new Layer(weights1, biases1);

		INDArray weights2 = Nd4j.create(new float[] { 10f, 10f }, new int[] { 1, 2 }, 'c');
		INDArray biases2 = Nd4j.create(new float[] { -18f }, new int[] { 1, 1 }, 'c');
		Layer layer2 = new Layer(weights2, biases2);

		NeuralNet net = new NeuralNet(layer1, layer2);

		System.out.println("x1=0.4, x2=0.6  -> y=" + net.forward(0.4f, 0.6f));
		System.out.println("x1=0.6, x2=0.2  -> y=" + net.forward(0.6f, 0.2f));
		System.out.println("x1=0.7, x2=0.7  -> y=" + net.forward(0.7f, 0.7f));
		System.out.println("x1=0.6, x2=0.51 -> y=" + net.forward(0.6f, 0.51f));

		Pair<INDArray, INDArray> pair = ScatterPlott.createFeaturesAndLabels(net);
		for (int i = 0; i < net.getOutputSize(); i++)
			ScatterPlott.plotTrainingData("Output", pair.getFirst(), pair.getSecond(), i);
	}

}
