package herrler.backprop;

import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class ErrorPlot extends ApplicationFrame{
	
	private ChartPanel jpanel;
	private XYSeries series;
	private JFreeChart jfreechart;
	int iteration = 0;
	ArrayList<Float> errorlist = new ArrayList<Float>();

	public ErrorPlot() {
		super("Training Error");
		XYSeriesCollection data = new XYSeriesCollection();
		jfreechart = ChartFactory.createXYLineChart("Training Error", "Iteration", "MSE", data,PlotOrientation.HORIZONTAL , true, false, false);
		jpanel = new ChartPanel(jfreechart);
		jpanel.setPreferredSize(new Dimension(400, 300));
		series = new XYSeries("Error");

		add(jpanel);
	}

	public static ErrorPlot openWindow() {
		ErrorPlot errorPlot = new ErrorPlot();
		errorPlot.pack();
		RefineryUtilities.centerFrameOnScreen(errorPlot);
		errorPlot.setVisible(true);
		return errorPlot;
	}

	public void add(float error) {
		iteration++;
		errorlist.add(error);
		
		if (errorlist.size()<20)
			return;
		
		float sum = 0;
		for (Float value : errorlist)
			sum+=value;
		sum/=errorlist.size();
		errorlist.clear();
		
		series.add(iteration, sum);
		XYSeriesCollection data = new XYSeriesCollection();
		data.addSeries(series);
		jfreechart = ChartFactory.createXYLineChart("Training Error", "Iteration", "MSE", data,PlotOrientation.VERTICAL , true, false, false);
		jpanel.setChart(jfreechart);
	}
}
