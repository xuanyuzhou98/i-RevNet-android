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
import org.nd4j.linalg.learning.config.Nesterovs;
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
                int[] nChannels = new int[]{16, 64, 256};
                int[] nBlocks = new int[]{18, 18, 18};
                int[] nStrides = new int[]{1, 2, 2};
                int channels = 3;
                int init_ds = 0;
                int in_ch = channels * (int) Math.pow(2, init_ds);
                int n = in_ch / 2;
                int outputNum = 10; // number of output classes
                final int numRows = 32;
                final int numColumns = 32;
                int rngSeed = 1234; // random number seed for reproducibility
                int numEpochs = 1; // number of epochs to perform
                Random randNumGen = new Random(rngSeed);
                int batchSize = 54; // batch size for each epoch
                int mult = 4;
                try {
                    ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                            .seed(rngSeed)
                            .activation(Activation.IDENTITY)
                            .updater(new Nesterovs(0.1, 0.9))
                            .weightInit(WeightInit.XAVIER)
                            .l1(1e-7)
                            .l2(5e-5)
                            .graphBuilder();
                    graph.addInputs("input").setInputTypes(InputType.convolutional(numRows, numColumns, channels)); //(3, 3, 32, 32)
                    String lastLayer = "input";
                    if (init_ds != 0) {
                        graph.addLayer("init_psi", new PsiLayer.Builder()
                                .BlockSize(init_ds)
                                .nIn(channels)
                                .nOut(in_ch)
                                .build(), "input");
                        lastLayer = "init_psi";
                    }

                    graph.addVertex("x0", new SubsetVertexN(n-1, 0), lastLayer) //(3, 1, 32, 32)
                            .addVertex("tilde_x0", new SubsetVertexN(in_ch-1, n), lastLayer); //(3, 2, 32, 32)
                    ActivationLayer relu = new ActivationLayer.Builder()
                            .activation(Activation.RELU)
                            .build();
                    graph.addLayer("firstRelu", relu, "tilde_x0");
                    int in_ch_Block = in_ch;
                    String input1 = "x0"; //(3, 1, 32, 32)
                    String input2 = "firstRelu"; // (3, 2, 32, 32)
                    List<IRevBlock> blockList = new ArrayList<>();
                    for (int i = 0; i < 3; i++) { // for each stage
                        for (int j = 0; j < nBlocks[i]; j++) { // for each block in the stage
                            int stride = 1;
                            if (j == 0) {
                                stride = nStrides[i];
                            }
                            IRevBlock innerIRevBlock = new IRevBlock(graph, in_ch_Block, nChannels[i], stride,
                                    mult, input1, input2, String.valueOf(i) + String.valueOf(j));
                            String[] outputs = innerIRevBlock.getOutput();
                            input1 = outputs[0];
                            input2 = outputs[1];
                            blockList.add(innerIRevBlock);
                            in_ch_Block = 2 * nChannels[i];
                        }
                    }

                    // layers
                    BatchNormalization BNlayer = new BatchNormalization.Builder()
                            .nIn(n * 4 * 2)
                            .nOut(n * 4 * 2)
                            .build();

                    ActivationLayer ReLulayer = new ActivationLayer.Builder()
                            .activation(Activation.RELU)
                            .build();

                    GlobalPoolingLayer Poolinglayer = new GlobalPoolingLayer.Builder()
                            .poolingType(PoolingType.AVG)
                            .build();

                    DenseLayer Denselayer = new DenseLayer.Builder()
                            .activation(Activation.IDENTITY)
                            .nOut(outputNum)
                            .build();

                    OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nOut(outputNum)
                            .activation(Activation.SOFTMAX)
                            .build();

                    graph.addVertex("merge", new MergeVertex(), input1, input2)
                            .addLayer("outputBN", BNlayer, "merge")
                            .addLayer("outputRelu", ReLulayer, "outputBN")
                            .addLayer("outputPool", Poolinglayer, "outputRelu")
                            .addLayer("outputProb", Denselayer, "outputPool")
                            .addLayer("output", outputLayer, "outputProb")
                            .setOutputs("output");

                    ComputationGraphConfiguration conf = graph.build();
                    ComputationGraph model = new ComputationGraph(conf);
                    model.init();
                    Log.d("Output", "start output");
                    INDArray[] TestArray = new INDArray[1];
                    INDArray sample = Nd4j.create(3, 32, 32, 3);
                    TestArray[0] = sample;
                    INDArray[] outputs = model.output(TestArray);
                    Log.d("Success!", "Success!!!!!!!!");
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