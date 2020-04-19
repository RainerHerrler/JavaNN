package herrler.feedforward;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Shape;
import java.util.Random;

import javax.swing.JPanel;

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

public class ScatterPlott extends ApplicationFrame {

	private static final long serialVersionUID = 1L;

	static Pair<INDArray, INDArray> createFeaturesAndLabels(NeuralNet net) {
		INDArray features = Nd4j.create(new float[] {}, 0, net.getInputSize());
		INDArray labels = Nd4j.create(new float[] {}, 0, net.getOutputSize());
		Nd4j.getRandom().setSeed(3000);
		for (int s = 0; s < 120; s++) {
			INDArray x = Nd4j.rand(net.getInputSize(), 1);
			features = Nd4j.concat(0, features, x.transpose());

			// System.out.println("x=" + x.transpose());
			INDArray y = net.forward(x);
			labels = Nd4j.concat(0, labels, y.transpose());
			// System.out.println("y=" + y.transpose());
			// System.out.println("");
		}
		return new Pair<INDArray, INDArray>(features, labels);
	}

	public static void plotTrainingData(String plotName, INDArray features, INDArray labels, int featureNr) {
		XYSeries positive = new XYSeries("y" + (featureNr + 1) + ">0.5");
		XYSeries negative = new XYSeries("y" + (featureNr + 1) + "<0.5");
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
		ScatterPlott scatterplotdemo4 = new ScatterPlott(plotName +  " " + (featureNr +1), xySeriesCollection,
				new Dimension(400, 300), featureNr);
		scatterplotdemo4.pack();
		RefineryUtilities.centerFrameOnScreen(scatterplotdemo4);
		scatterplotdemo4.setVisible(true);
	}

	public ScatterPlott(String title, XYDataset data, Dimension dimension, int shapeType) {
		super(title);
		JPanel jpanel = createDemoPanel(title, data, shapeType);
		jpanel.setPreferredSize(dimension);
		add(jpanel);
	}

	public static JPanel createDemoPanel(String title, XYDataset data, int shapeType) {
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

		return new ChartPanel(jfreechart);
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
		ScatterPlott scatterplotdemo4 = new ScatterPlott("Scatter Plot Demo 4", samplexydataset2(),
				new Dimension(640, 480), 0);
		scatterplotdemo4.pack();
		RefineryUtilities.centerFrameOnScreen(scatterplotdemo4);
		scatterplotdemo4.setVisible(true);
	}

}
