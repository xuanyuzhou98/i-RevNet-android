package com.example.mnist;

import android.os.AsyncTask;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.File;
import java.lang.Math;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class MainActivity extends AppCompatActivity {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/mnist";
    private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute("");
                ProgressBar bar = findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });
    }

    private class AsyncTaskRunner extends AsyncTask<String, Integer, String> {
        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
        }

        // This is our main background thread for the neural net
        @Override
        protected String doInBackground(String... params) {
            int channels = 1;
            int init_ds = 2;
            int in_ch = channels * (int)Math.pow(2, init_ds);
            int n = in_ch / 2;
            int outputNum = 10; // number of output classes
            boolean first = true;
            final int numRows = 28;
            final int numColumns = 28;
            int rngSeed = 1234; // random number seed for reproducibility
            int numEpochs = 1; // number of epochs to perform
            Random randNumGen = new Random(rngSeed);
            int batchSize = 54; // batch size for each epoch
            int mult = 4;
            try {
                if (!new File(basePath + "/mnist_png").exists()) {
                    Log.d("Data download", "Data downloaded from " + dataUrl);
                    String localFilePath = basePath + "/mnist_png.tar.gz";
                    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
                        DataUtilities.extractTarGz(localFilePath, basePath);
                    }
                }
                // vectorization of train data
                File trainData = new File(basePath + "/mnist_png/training");
                FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
                ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                trainRR.initialize(trainSplit);
                DataSetIterator mnistTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
                // pixel values from 0-255 to 0-1 (min-max scaling)
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.fit(mnistTrain);
                mnistTrain.setPreProcessor(scaler);

                // vectorization of test data
                File testData = new File(basePath + "/mnist_png/testing");
                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                testRR.initialize(testSplit);
                DataSetIterator mnistTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
                mnistTest.setPreProcessor(scaler); // same normalization for better results
                Log.d("build model", "Build model....");

                Map<Integer, Double> learningRateSchedule = new HashMap<>();
                learningRateSchedule.put(0, 0.06);
                learningRateSchedule.put(200, 0.05);
                learningRateSchedule.put(600, 0.028);
                learningRateSchedule.put(800, 0.0060);
                learningRateSchedule.put(1000, 0.001);

                ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                        .seed(1234)
                        .activation(Activation.IDENTITY)
                        .updater(new Sgd(0.05))
                        .weightInit(WeightInit.XAVIER)
                        .l1(1e-7)
                        .l2(5e-5)
                        .graphBuilder();
                graph.addInputs("input").setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                        .addLayer("init_psi", new PsiLayer.Builder()
                                .BlockSize(init_ds)
                                .nIn(channels)
                                .nOut(in_ch)
                                //.outWidth()
                                .build(), "input")
                        .addVertex("x0", new SubsetVertex(0, n-1), "init_psi")
                        .addVertex("tilde_x0", new SubsetVertex(n, in_ch-1), "init_psi");

                String[] output = iRevBlock(graph, n, n * 4, 2, first, 0,
                        mult, "x0", "tilde_x0", "irev1");
                OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build();

                graph.addVertex("merge", new MergeVertex(), output[0], output[1])
                        .addLayer("outputBN", new BatchNormalization.Builder()
                                .nIn(n * 4 * 2)
                                .nOut(n * 4 * 2)
                                .build(), "merge")
                        .addLayer("outputRelu", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                "outputBN")
                        .addLayer("outputPool", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(),
                                "outputRelu")
                        .addLayer("outputProb", new DenseLayer.Builder().activation(Activation.RELU)
                                .nOut(outputNum).build(), "outputPool")
                        .addLayer("output", outputLayer, "outputProb")
                        .setOutputs("output");

                ComputationGraphConfiguration conf = graph.build();
                ComputationGraph model = new ComputationGraph(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(10));
                Log.d("Total num of params", "Total num of params" + model.numParams());

                Log.d("train model", "Train model....");
                for(int l=0; l<=numEpochs; l++) {
                    model.fit(mnistTrain);
                }

                Log.d("evaluate model", "Evaluate model....");
                Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
                while(mnistTest.hasNext()){
                    DataSet next = mnistTest.next();
                    INDArray testOutput = model.output(next.getFeatures())[0]; //get the networks prediction
                    eval.eval(next.getLabels(), testOutput); //check the prediction against the true class
                }
            }catch(Exception e){
                e.printStackTrace();
            }
            return "";
        }

        protected String bottleneckBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                                    int in_ch, int out_ch, int stride, boolean first, float dropout_rate,
                                    int mult, String input, String prefix) {
            if (!first) {
                graphBuilder
                        .addLayer(prefix + "_bn0", new BatchNormalization.Builder()
                                .nIn(out_ch / mult)
                                .nOut(out_ch / mult)
                                .build(), input)
                        .addLayer(prefix + "_act0", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                prefix + "_bn0");
                input = prefix + "_act0";
            }
            graphBuilder
                    .addLayer(prefix+"_conv1", new ConvolutionLayer.Builder(3, 3)
                            .nIn(in_ch)
                            .stride(stride, stride)
                            .padding(1, 1)
                            .nOut(out_ch/mult)
                            .build(), input)
                    .addLayer(prefix+"_bn1", new BatchNormalization.Builder()
                            .nIn(out_ch/mult)
                            .nOut(out_ch/mult)
                            .build(), prefix+"_conv1")
                    .addLayer(prefix+"_act1", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                            prefix+"_bn1")
                    .addLayer(prefix+"_conv2", new ConvolutionLayer.Builder(3, 3)
                            .nIn(out_ch/mult)
                            .padding(1, 1)
                            .nOut(out_ch/mult)
                            .build(), prefix + "_act1")
                    .addLayer(prefix+"_drop2", new DropoutLayer.Builder(1-dropout_rate).build(),
                            prefix + "_conv2")
                    .addLayer(prefix+"_bn2", new BatchNormalization.Builder()
                            .nIn(out_ch / mult)
                            .nOut(out_ch / mult)
                            .build(), prefix + "_drop2")
                    .addLayer(prefix+"_act2", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                            prefix + "_bn2")
                    .addLayer(prefix, new ConvolutionLayer.Builder(3, 3)
                            .nIn(out_ch / mult)
                            .padding(1, 1)
                            .nOut(out_ch)
                            .build(), prefix + "_act2");
            return prefix;
        }

        protected String[] iRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                                    int in_ch, int out_ch, int stride, boolean first, float dropout_rate,
                                    int mult, String input1, String input2, String prefix) {
            String Fx2 = bottleneckBlock(graphBuilder, in_ch, out_ch, stride, first, dropout_rate,
                            mult, input2, prefix + "_btnk");
            if (stride == 2) {
                graphBuilder
                        .addLayer(prefix + "_psi1", new PsiLayer.Builder()
                                .BlockSize(stride)
                                .nIn(in_ch)
                                .nOut(out_ch)
                                .build(), input1)
                        .addLayer(prefix + "_psi2", new PsiLayer.Builder()
                                .BlockSize(stride)
                                .nIn(in_ch)
                                .nOut(out_ch)
                                .build(), input2);
                input1 = prefix + "_psi1";
                input2 = prefix + "_psi2";
            }
            graphBuilder
                    .addVertex(prefix + "_y1", new ElementWiseVertex(ElementWiseVertex.Op.Add), Fx2, input1);
            String[] output = new String[2];
            output[0] = input2;
            output[1] = prefix + "_y1";
            return output;
        }

        //This is called from background thread but runs in UI for a progress indicator
        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);


            //Hide the progress bar now that we are finished
            ProgressBar bar = findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
        }
    }
}