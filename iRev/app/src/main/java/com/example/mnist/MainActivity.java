package com.example.mnist;

import android.os.AsyncTask;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;

import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ser.impl.IteratorSerializer;


import java.io.File;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class MainActivity extends AppCompatActivity {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/mnist";
    private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
    private long bottleNeckForwardFLOPS = 0;
    private long bottleNeckBackwardFLOPS = 0;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                AsyncTaskRunner runner = new AsyncTaskRunner();
//                runner.execute("");
                ProgressBar bar = findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
                new Thread(new Task()).start();
            }
        });
    }

    private class Task implements Runnable {
        @Override
        public void run() {
            try {
//                    Thread.sleep(1000);
                int[] nChannels = new int[]{16, 64, 256};
                int[] nBlocks = new int[]{18, 18, 18};
                int[] nStrides = new int[]{1, 2, 2};
                int channels = 1;
                int init_ds = 2;
                int in_ch = channels * (int) Math.pow(2, init_ds);
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
//            INDArray sample = Nd4j.ones(1, 3, 28, 28);
                try {
//                        if (!new File(basePath + "/mnist_png").exists()) {
//                            Log.d("Data download", "Data downloaded from " + dataUrl);
//                            String localFilePath = basePath + "/mnist_png.tar.gz";
//                            if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
//                                DataUtilities.extractTarGz(localFilePath, basePath);
//                            }
//                        }
//                        // vectorization of train data
//                        File trainData = new File(basePath + "/mnist_png/training");
//                        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//                        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
//                        ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
//                        trainRR.initialize(trainSplit);
//                        DataSetIterator mnistTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
//                        // pixel values from 0-255 to 0-1 (min-max scaling)
//                        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//                        scaler.fit(mnistTrain);
//                        mnistTrain.setPreProcessor(scaler);
//
//                        // vectorization of test data
//                        File testData = new File(basePath + "/mnist_png/testing");
//                        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//                        ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
//                        testRR.initialize(testSplit);
//                        DataSetIterator mnistTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
//                        mnistTest.setPreProcessor(scaler); // same normalization for better results
//                        Log.d("build model", "Build model....");
//
//                        Map<Integer, Double> learningRateSchedule = new HashMap<>();
//                        learningRateSchedule.put(0, 0.06);
//                        learningRateSchedule.put(200, 0.05);
//                        learningRateSchedule.put(600, 0.028);
//                        learningRateSchedule.put(800, 0.0060);
//                        learningRateSchedule.put(1000, 0.001);

                    // Count FLOPS
                    ConvolutionLayer PSIlayer = new PsiLayer.Builder()
                            .BlockSize(init_ds)
                            .nIn(channels)
                            .nOut(in_ch)
                            //.outWidth()
                            .build();
                    // TODO: verify PSIlayer shape.
//                int[] PSIlayerShape = getConvLayerOutShape(numRows, numColumns, 3, 1);
//                int PSIlayerChannels = in_ch;

                    ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                            .seed(1234)
                            .activation(Activation.IDENTITY)
                            .updater(new Sgd(0.05))
                            .weightInit(WeightInit.XAVIER)
                            .l1(1e-7)
                            .l2(5e-5)
                            .graphBuilder();


                    int in_ch_Block = in_ch;
                    List<IRevBlock> blockList = new ArrayList<>();
                    for (int i = 0; i < 3; i++) { // for each stage
                        for (int j = 0; j < nBlocks[i]; j++) { // for each block in the stage
                            // create new IRevBlock
                            IRevBlock innerIRevBlock = new IRevBlock(graph, in_ch_Block, nChannels[i], nStrides[i],first, mult, String input1, String input2, String prefix);
                            // append to list
                            blockList.add(innerIRevBlock);
                            // update
                            in_ch_Block = 2 * nChannels[i];
                            first = false;
                        }
                    }


                    graph.addInputs("input").setInputTypes(InputType.convolutionalFlat(28, 28, 3));
//                            .addLayer("init_psi", PSIlayer, "input")
//                            //.addVertex("x0", new SubsetVertex(0, n - 1), "init_psi")
//                            .addVertex("tilde_x0", new SubsetVertex(n, in_ch - 1), "init_psi");

                    // layers
                    BatchNormalization BNlayer = new BatchNormalization.Builder()
                            .nIn(n * 4 * 2)
                            .nOut(n * 4 * 2)
                            .build();
//                int[] BNlayerShape = PSIlayerShape;
                    int BNlayerChannels = n * 4 * 2;

                    ActivationLayer ReLulayer = new ActivationLayer.Builder()
                            .activation(Activation.RELU)
                            .build();
//                int[] ReLulayerShape = BNlayerShape;
                    int ReLulayerChannels = BNlayerChannels;

                    GlobalPoolingLayer Poolinglayer = new GlobalPoolingLayer.Builder()
                            .poolingType(PoolingType.AVG)
                            .build();
//                int[] poolOutShape = ReLulayerShape;
                    int poolOutChannel = ReLulayerChannels;

                    DenseLayer Denselayer = new DenseLayer.Builder().
                            activation(Activation.RELU)
                            .nOut(outputNum)
                            .build();
//                    int fcInShape = 1 * 16 * poolOutChannel;
//                    long flopFC = getFlopCountFC(fcInShape, outputNum);
//                    long flopFCBack = getFlopCountFCBackward(fcInShape, outputNum);
//                    Log.d("FC Flop count", "forward count " + flopFC);
//                    Log.d("FC Flop count", "backward count " + flopFCBack);
//
//                    long totalForwardFLOPS = flopFC + bottleNeckForwardFLOPS;
//                    long totalBackwardFLOPS = flopFCBack + bottleNeckBackwardFLOPS;
//                    Log.d("TOTAL Flop count", "forward count " + totalForwardFLOPS);
//                    Log.d("TOTAL Flop count", "backward count " + totalBackwardFLOPS);

                    IRevBlock irev1 = new IRevBlock(graph, 3, 6, 1, first,
                            mult, "x0", "input", "irev1");
                    String[] output = irev1.getOutput();
                    graph.setOutputs(output[0]);
//                    OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                            .nOut(outputNum)
//                            .activation(Activation.SOFTMAX)
//                            .build();
//
//                    graph.addVertex("merge", new MergeVertex(), output[0], output[1])
//                            .addLayer("outputBN", BNlayer, "merge")
//                            .addLayer("outputRelu", ReLulayer, "outputBN")
//                            .addLayer("outputPool", Poolinglayer, "outputRelu")
//                            .addLayer("outputProb", Denselayer, "outputPool")
//                            .addLayer("output", outputLayer, "outputProb")
//                            .setOutputs("output", "outputPool");


                    ComputationGraphConfiguration conf = graph.build();
                    ComputationGraph model = new ComputationGraph(conf);
                    model.init();
                    Log.d("Output", "start output");
                    INDArray[] TestArray = new INDArray[1];
                    INDArray sample = Nd4j.create(3, 3, 28, 28);
                    TestArray[0] = sample;
                    INDArray[] outputs = model.output(TestArray);
                    INDArray[] gradients = irev1.bottleneck.gradient(Nd4j.create(3, 3, 28, 28),
                            Nd4j.create(1, 24, 7, 7));//Test Gradient method
                    Log.d("Success!", gradients.toString());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

    }


    private long getFlopCountConv(int channels, int filter_size, int num_filters,
                                  int outShapeH, int outShapeW) {
        return (2 * channels * filter_size * filter_size - 1) * num_filters * outShapeH * outShapeW;
    }

    private long getFlopCountFC(int inputSize, int outputSize) {
        return (2 * inputSize - 1) * outputSize;
    }

    private long getFlopCountConvBackward(int channels, int filter_size, int num_filters,
                                          int outShapeH, int outShapeW) {
        int out = outShapeH * outShapeW;
        int db = out;
        int dw = num_filters * ((2 * out - 1) * channels * filter_size * filter_size);
        int dx_cols = channels * filter_size * filter_size * (2 * num_filters - 1) * out;
        int dx = channels * filter_size * filter_size * out;
        return db + dw + dx_cols + dx;
    }

    private long getFlopCountFCBackward(int inputSize, int outputSize) {
        int db = outputSize;
        int dx = (2 * outputSize - 1) * inputSize;
        int dw = inputSize * outputSize;
        return db + dx + dw;
    }
}