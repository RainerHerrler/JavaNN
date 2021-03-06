package herrler.backprop;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Shape;
import java.util.ArrayList;
import java.util.Random;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.ShapeUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kotlin.Pair;

public class ScatterPlot extends ApplicationFrame {

	private static final long serialVersionUID = 1L;
	private static ChartPanel jpanel;
	private static ArrayList<INDArray> allX;

	/**
	 * The number of examples to create
	 * 
	 * @param net
	 * @param examples
	 * @return
	 */
	static Pair<INDArray, INDArray> createFeaturesAndLabels(NeuralNet net, int examples) {
		INDArray features = Nd4j.create(new float[] {}, 0, net.getInputSize());
		INDArray labels = Nd4j.create(new float[] {}, 0, net.getOutputSize());
		if (allX == null) {
			Nd4j.getRandom().setSeed(3000);
			allX = new ArrayList<INDArray>();
			for (int s = 0; s < examples; s++) {
				allX.add(Nd4j.rand(net.getInputSize(), 1));
			}
		}

		for (INDArray x : allX) {
			features = Nd4j.concat(0, features, x.transpose());

			INDArray y = net.forward(x);
			labels = Nd4j.concat(0, labels, y.transpose());
		}
		return new Pair<INDArray, INDArray>(features, labels);
	}

	static Pair<INDArray, INDArray> createFeaturesAndLabels2(NeuralNet net, int examples) {
		INDArray features = Nd4j.create(new float[] {}, 0, net.getInputSize());
		INDArray labels = Nd4j.create(new float[] {}, 0, net.getOutputSize());
		Nd4j.getRandom().setSeed(3000);
		for (int x1 = 0; x1 < examples; x1++) {
			for (int x2 = 0; x2 < examples; x2++) {
				float erg1 = x1 * 0.1f / examples;
				float erg2 = x2 * 0.1f / examples;
				INDArray x = Nd4j.rand(net.getInputSize(), 1);
				Nd4j.create(new float[] { erg1, erg2 }, net.getInputSize(), 1);
				features = Nd4j.concat(0, features, x.transpose());

				INDArray y = net.forward(x);
				labels = Nd4j.concat(0, labels, y.transpose());
			}
		}
		return new Pair<INDArray, INDArray>(features, labels);
	}

	public static ScatterPlot plotTrainingData(String plotName, INDArray features, INDArray labels, int featureNr,
			String outputvalName) {
		XYSeriesCollection xySeriesCollection = createXYSeries(features, labels, featureNr, outputvalName);
		ScatterPlot scatterplotdemo4 = new ScatterPlot(plotName + " " + (featureNr + 1), xySeriesCollection,
				new Dimension(400, 300), featureNr);
		scatterplotdemo4.pack();
		RefineryUtilities.centerFrameOnScreen(scatterplotdemo4);
		scatterplotdemo4.setVisible(true);
		return scatterplotdemo4;
	}

	private static XYSeriesCollection createXYSeries(INDArray features, INDArray labels, int featureNr,
			String outputvalName) {
		XYSeries positive = new XYSeries(outputvalName + (featureNr + 1) + ">0.5");
		XYSeries negative = new XYSeries(outputvalName + (featureNr + 1) + "<0.5");
		for (int i = 0; i < labels.size(0); i++) {
			float x = features.getFloat(i, 0);
			float y = features.getFloat(i, 1);
			if (labels.getFloat(i, featureNr) > 0.5f)
				positive.add(x, y);
			else
				negative.add(x, y);
		}
		XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
		xySeriesCollection.addSeries(positive);
		xySeriesCollection.addSeries(negative);
		return xySeriesCollection;
	}

	public ScatterPlot(String title, XYDataset data, Dimension dimension, int shapeType) {
		super(title);
		JFreeChart jfreechart = createFreeChart(title, data, shapeType);
		jpanel = new ChartPanel(jfreechart);
		jpanel.setPreferredSize(dimension);
		add(jpanel);
	}

	public void update(String plotName, INDArray features, INDArray labels, int featureNr, String outputvalName) {
		XYSeriesCollection xySeriesCollection = createXYSeries(features, labels, featureNr, outputvalName);
		update(plotName, xySeriesCollection, featureNr);
	}

	public void update(String title, XYDataset data, int shapeType) {
		JFreeChart jfreechart = createFreeChart(title, data, shapeType);
		jpanel.setChart(jfreechart);
	}

	private static JFreeChart createFreeChart(String title, XYDataset data, int shapeType) {
		JFreeChart jfreechart = ChartFactory.createScatterPlot(title, "X1", "X2", data, PlotOrientation.VERTICAL, true,
				true, false);
		Shape shape = new Shape[] { //
				ShapeUtilities.createDiagonalCross(3, 0.2f), //
				ShapeUtilities.createDiamond(4), //
				ShapeUtilities.createUpTriangle(4)//
		}[shapeType];

		XYPlot xyPlot = (XYPlot) jfreechart.getPlot();
		xyPlot.setDomainCrosshairVisible(true);
		xyPlot.setRangeCrosshairVisible(true);
		XYItemRenderer renderer = xyPlot.getRenderer();
		renderer.setSeriesShape(0, shape);
		renderer.setSeriesPaint(0, Color.red);
		renderer.setSeriesShape(1, shape);
		renderer.setSeriesPaint(1, Color.blue);
		return jfreechart;
	}

	private static XYDataset samplexydataset2() {
		int cols = 20;
		int rows = 20;
		XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
		xySeriesCollection.addSeries(createSeries(cols / 2, rows));
		xySeriesCollection.addSeries(createSeries(cols / 2, rows));
		return xySeriesCollection;
	}

	private static XYSeries createSeries(int cols, int rows) {
		double[][] values = new double[cols][rows];
		XYSeries series = new XYSeries("Random");
		Random rand = new Random();
		for (int i = 0; i < values.length; i++) {
			for (int j = 0; j < values[i].length; j++) {
				double x = rand.nextGaussian();
				double y = rand.nextGaussian();
				series.add(x, y);
			}
		}
		return series;
	}

	public static void main(String args[]) {
		ScatterPlot scatterplotdemo4 = new ScatterPlot("Scatter Plot Demo 4", samplexydataset2(),
				new Dimension(640, 480), 0);
		scatterplotdemo4.pack();
		RefineryUtilities.centerFrameOnScreen(scatterplotdemo4);
		scatterplotdemo4.setVisible(true);
	}

	public static ScatterPlot openWindow() {
		return ScatterPlot.plotTrainingData("Waiting", Nd4j.create(0), Nd4j.create(0), 0, "a");
	}

	public static void showOutputs(ScatterPlot plot, String plotName, NeuralNet net) {
		Pair<INDArray, INDArray> pair = ScatterPlot.createFeaturesAndLabels(net, 120);
		// Pair<INDArray, INDArray> pair = ScatterPlot.createFeaturesAndLabels(net,
		// 120);
		for (int i = 0; i < net.getOutputSize(); i++)
			plot.update(plotName, pair.getFirst(), pair.getSecond(), i, "a");
	}

	public void showOutputs(String plotName, NeuralNet net) {
		showOutputs(this, plotName, net);
	}

}
