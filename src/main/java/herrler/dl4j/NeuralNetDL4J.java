/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package herrler.dl4j;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Example: Train a network to reproduce certain mathematical functions, and
 * plot the results. Plotting of the network output occurs every 'plotFrequency'
 * epochs. Thus, the plot shows the accuracy of the network predictions as
 * training progresses. A number of mathematical functions are implemented here.
 * Note the use of the identity function on the network output layer, for
 * regression
 *
 * @author Alex Black
 */
public class NeuralNetDL4J {

	// Random number generator seed, for reproducability
	public static final int seed = 12345;
	// Number of epochs (full passes of the data)
	public static final int nEpochs = 2000;
	// How frequently should we plot the network output?
	// Number of data points
	private static final int nSamples = 1000;
	// Batch size: i.e., each epoch has nSamples/batchSize parameter updates
	public static final int batchSize = 100;
	// Network learning rate
	public static final double learningRate = 0.1;
	public static final Random rng = new Random(seed);

	public static void main(final String[] args) {		
		final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()//
		.seed(seed)//
		.weightInit(WeightInit.XAVIER)//
		.updater(new Nesterovs(learningRate, 0.9)).list()//
		.layer(new DenseLayer.Builder().nIn(2).nOut(3).activation(Activation.RELU).build())
		.layer(new DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.RELU).build())
		.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)//
				.activation(Activation.SIGMOID).nIn(3).nOut(1).build())
		.build();

		// Generate the training data
		final DataSetIterator iterator = getTrainingData(nSamples, batchSize, rng);

		// Create the network
		final MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		// Train the network on the full data set, and evaluate in periodically
		ScatterPlott window = ScatterPlott.openWindow();
		for (int i = 0; i < nEpochs; i++) {
			iterator.reset();
			net.fit(iterator);
			window.showOutputs("Epoch " + i, net);
		}
	}

	/**
	 * Create a DataSetIterator for training
	 * 
	 * @param x         X values
	 * @param function  Function to evaluate
	 * @param batchSize Batch size (number of examples for every call of
	 *                  DataSetIterator.next())
	 * @param rng       Random number generator (for repeatability)
	 */
	private static DataSetIterator getTrainingData(final int sampleSize, int batchSize, final Random rng) {
		INDArray features = Nd4j.create(sampleSize, 2);
		INDArray labels = Nd4j.create(sampleSize, 1);
		
		for (int i = 0; i < sampleSize; i++) {
			INDArray x = Nd4j.rand(1, 2); // create random 2d verctor as a sample
			features.putRow(i, x);
			float expected = (x.getFloat(0) > 0.5f && x.getFloat(1) > 0.5f) ? 1f : 0f;
			// float expected = (x.getFloat(0) > 0.5f) ? 1f : 0f;
			INDArray y = Nd4j.create(new float[] { expected }, new int[] { 1, 1 }, 'c');
			labels.putRow(i, y);
		}
		DataSet newData = new DataSet(features, labels);

		return new ListDataSetIterator<>(newData.asList(), batchSize);
	}
}
