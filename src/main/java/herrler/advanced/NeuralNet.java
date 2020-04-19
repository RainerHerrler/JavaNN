package herrler.advanced;

import java.awt.Dimension;
import java.util.ArrayList;

import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import kotlin.Pair;

public class NeuralNet {

	final Layer[] layers;
	float learningRate = 0.001f;

	public NeuralNet(Layer... layers) {
		this.layers = layers;
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

	public float fit(INDArray x, INDArray yGiven) {
		INDArray y = forward(x);
		float e1 = calcMSEError(yGiven, y);
		INDArray localGradients = calcMSEGradients(yGiven, y);
		for (int i = layers.length - 1; i >= 0; i--) {
			layers[i].backprop(localGradients, learningRate);
		}
		INDArray y2 = forward(x);
		float e2 = calcMSEError(yGiven, y2);
		System.out.println("Errordiff :" + e1 + " -> " + e2 + " = " + Math.signum(e1 - e2));
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
		NeuralNet net1 = createExampleNet2();
		int inputSize = net1.getInputSize();
		int outputSize = net1.getOutputSize();

		NeuralNet net2 = new NeuralNet(new Layer(inputSize, outputSize));
		net2.learningRate = 0.1f;

		for (int s = 0; s < 10000; s++) {
			INDArray x = Nd4j.rand(inputSize, 1); // create random sample
			INDArray yGround = net1.forward(x); // result of reference net

			net2.fit(x, Transforms.round(yGround));
		}
		net2.layers[0].printParameters(); // print resulting net

		Pair<INDArray, INDArray> pair = ScatterPlott.createFeaturesAndLabels(net2);
		for (int i = 0; i < outputSize; i++)
			ScatterPlott.plotTrainingData("Feature", pair.getFirst(), pair.getSecond(), i);
	}
	

	private static NeuralNet createExampleNet1() {
		Layer layer = new Layer(2, 1);
		layer.weights = Nd4j.create(new float[] { 0, 1 }, new int[] { 1, 2 }, 'c');
		layer.biases = Nd4j.create(new float[] { -0.5f }, new int[] { 1, 1 }, 'c');
		return new NeuralNet(layer);
	}
	
	private static NeuralNet createExampleNet2() {
		Layer layer = new Layer(2, 2);
		layer.weights = Nd4j.create(new float[] { 0, 1, 1, 0 }, new int[] { 2, 2 }, 'c');
		layer.biases = Nd4j.create(new float[] { -0.5f, -0.5f }, new int[] { 2, 1 }, 'c');
		return new NeuralNet(layer);
	}


	private static NeuralNet createExampleNet3() {
		Layer layer = new Layer(2, 3);
		layer.weights = Nd4j.create(new float[] { 0, 1, 1, 0, 0.5f, 0.5f }, new int[] { 3, 2 }, 'c');
		layer.biases = Nd4j.create(new float[] { -0.5f, -0.5f, -0.5f }, new int[] { 3, 1 }, 'c');
		return new NeuralNet(layer);
	}



}
