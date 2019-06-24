package com.example.cifar;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Environment;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.bytedeco.opencv.presets.opencv_core;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.RandomCropTransform;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Trainable;
import org.deeplearning4j.nn.updater.LayerUpdater;
import org.deeplearning4j.nn.updater.UpdaterBlock;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.StandardizeStrategy;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.NesterovsUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.learning.config.Nesterovs;

import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.tensorflow.framework.DebugOptions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/mnist";
    private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
    private static final boolean manual_gradients = true;
    private static final boolean half_precision = false;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        verifyStoragePermission(MainActivity.this);
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");


        if (half_precision) {
            Nd4j.setDefaultDataTypes(DataType.HALF, DataType.HALF);
        }

        System.out.println("ND4J Data Type Setting: " + Nd4j.dataType());

        final Button button = findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute("");
                ProgressBar bar = findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                button.performClick();
            }
        }, 1000);
    }

    public static void verifyStoragePermission(Activity activity) {
        // Get permission status
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission we request it
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
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
            try{
                int[] nChannels = new int[]{16, 64, 256};
                int[] nBlocks = new int[]{18, 18, 18};
                int[] nStrides = new int[]{1, 2, 2};
                int channels = 3;
                int init_ds = 0;
                int in_ch = channels * (int)Math.pow(2, init_ds);
                int n = in_ch / 2;
                int outputNum = 10; // number of output classes
                final int numRows = 32;
                final int numColumns = 32;
                int rngSeed = 1234; // random number seed for reproducibility
                int numEpochs = 10; // number of epochs to perform
                int batchSize = 100;
                int mult = 4;
                double init_lr = 0.01;
                // learning rate schedule
                Map<Integer, Double> learningRateSchedule = new HashMap<>();
                learningRateSchedule.put(0, init_lr);
                learningRateSchedule.put(60, init_lr * Math.pow(0.2, 1));
                learningRateSchedule.put(120, init_lr * Math.pow(0.2, 2));
                learningRateSchedule.put(160, init_lr * Math.pow(0.2, 3));
                // download data
                File baseDir = new File(basePath);
                if (!baseDir.exists()) {
                    baseDir.mkdir();
                }
                if (!new File(basePath + "/mnist_png").exists()) {
                    Log.d("Data download", "Data downloaded from " + dataUrl);
                    String localFilePath = basePath + "/mnist_png.tar.gz";
                    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
                        DataUtilities.extractTarGz(localFilePath, basePath);
                    }
                }
                // load data
                Random randNumGen = new Random(rngSeed);
                File trainData = new File(basePath + "/mnist_png/training");
                FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
                ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                trainRR.initialize(trainSplit);
                DataSetIterator mnistTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
                File testData = new File(basePath + "/mnist_png/testing");
                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                testRR.initialize(testSplit);
                DataSetIterator mnistTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
                // data preprocessing
                DataNormalization normalizer = new NormalizerStandardize();
                normalizer.fit(mnistTrain);
                mnistTrain.setPreProcessor(normalizer);
                mnistTest.setPreProcessor(normalizer);

                NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
                        .seed(rngSeed)
                        .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION,
                                learningRateSchedule), 0.9))
                        .weightDecay(5e-4);
                if (half_precision) {
                    config.dataType(DataType.HALF);
                }
                ComputationGraphConfiguration.GraphBuilder graph = config.graphBuilder();

                graph.addInputs("input").setInputTypes(InputType.convolutional(numRows, numColumns, channels));
                String lastLayer = "input";
                if (init_ds != 0) {
                    graph.addLayer("init_psi", new PsiLayer.Builder()
                            .BlockSize(init_ds)
                            .build(), "input");
                    lastLayer = "init_psi";
                }

                graph.addVertex("x0", new SubsetVertexN(0, n - 1), lastLayer)
                        .addVertex("tilde_x0", new SubsetVertexN(n, in_ch - 1), lastLayer);
                int in_ch_Block = in_ch;
                String input1 = "x0";
                String input2 = "tilde_x0";
                boolean first = true;
                List<IRevBlock> blockList = new ArrayList<>();
                for (int i = 0; i < 3; i++) { // for each stage
                    for (int j = 0; j < nBlocks[i]; j++) { // for each block in the stage
                        int stride = 1;
                        if (j == 0) {
                            stride = nStrides[i];
                        }
                        IRevBlock innerIRevBlock = new IRevBlock(graph, in_ch_Block, nChannels[i], stride, first,
                                mult, input1, input2, String.valueOf(i) + String.valueOf(j));
                        String[] outputs = innerIRevBlock.getOutput();
                        input1 = outputs[0];
                        input2 = outputs[1];
                        blockList.add(innerIRevBlock);
                        in_ch_Block = 2 * nChannels[i];
                        first = false;
                    }
                }

                ProbLayer probLayer = new ProbLayer(nChannels[nChannels.length - 1] * 2, outputNum);

                LossLayer lossLayer = new LossLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build();

                graph.addVertex("merge", new MergeVertex(), input1, input2)
                        .addLayer("outputProb", probLayer,"merge")
                        .addLayer("output", lossLayer, "outputProb")
                        .setOutputs( "output", "merge");

                ComputationGraphConfiguration conf = graph.build();
                ComputationGraph model = new ComputationGraph(conf);
                model.init();
                MemoryManager mg = Nd4j.getMemoryManager();
                mg.togglePeriodicGc(true);

                Log.d("Output", "start training");
                if (manual_gradients) {
                    int i = 0;
                    Gradient gradient = new DefaultGradient();
                    ComputationGraphUpdater updater = model.getUpdater();
                    for (int epoch = 0; epoch < numEpochs; epoch++) {
                        while (mnistTrain.hasNext()) {
                            Log.d("Iteration", "Running iter " + i);
                            DataSet data = mnistTrain.next();
                            INDArray label = data.getLabels();
                            INDArray features = data.getFeatures();

                            // Forward Pass
                            long StartTime = System.nanoTime();
                            INDArray[] outputs = model.output(false, false, features);
                            INDArray output = outputs[0];
                            INDArray merge = outputs[1];
                            long EndTime = System.nanoTime();
                            double elapsedTimeInSecond = (double) (EndTime - StartTime) / 1_000_000_000;
                            Log.d("forward time", String.valueOf(elapsedTimeInSecond));
                            Log.d("output", "finished forward iter " + i);

                            // Backward Pass
                            StartTime = System.nanoTime();
                            INDArray[] outputGradients = probLayer.gradient(merge, label);
                            INDArray dwGradient = outputGradients[1];
                            INDArray dbGradient = outputGradients[2];
                            INDArray[] lossGradient = Utils.splitHalf(outputGradients[0]);
                            INDArray[] hiddens = Utils.splitHalf(merge);
                            computeGradient(gradient, hiddens[0], hiddens[1],
                                    nBlocks, blockList, lossGradient);
                            gradient.setGradientFor("outputProb_denseWeight", dwGradient);
                            gradient.setGradientFor("outputProb_denseBias", dbGradient);
                            updater.update(gradient, i, epoch, batchSize, LayerWorkspaceMgr.noWorkspaces());
                            model.params().subi(gradient.gradient());
                            EndTime = System.nanoTime();
                            elapsedTimeInSecond = (double) (EndTime - StartTime) / 1_000_000_000;
                            Log.d("backward time", String.valueOf(elapsedTimeInSecond));
                            Log.d("output", "finished backward iter " + i);

                            // Evaluation
                            Evaluation eval = new Evaluation(10);
                            eval.eval(label, output);
                            Log.d("accuracy", eval.stats());
                            i++;
                        }
                    }
                } else {
                    model.fit(mnistTrain, numEpochs);
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            return "";
        }

        // This function computes the total gradient of the graph without referring to the stored activation
        protected void computeGradient(Gradient gradient, INDArray y1, INDArray y2, int[] nBlocks,
                                                            List<IRevBlock> blockList, INDArray[] lossGradient) {
            INDArray dy1 = lossGradient[0];
            INDArray dy2 = lossGradient[1];
            int cnt = blockList.size() - 1;
            Pair<String, INDArray>[] grads = new Pair[blockList.size() * 3];
            // from the last layer to the first layer
            for (int i = nBlocks.length - 1; i >= 0; i -= 1) { // for each stage
                for (int j = nBlocks[i] - 1; j >= 0; j -= 1) { // for each iRevBlock
                    IRevBlock iRev = blockList.get(cnt);
                    INDArray[] x = iRev.inverse(y1, y2);
                    INDArray x1 = x[0];
                    INDArray x2 = x[1];
                    Runtime.getRuntime().gc();
                    y1 = x1;
                    y2 = x2;
                    List<INDArray> gradients = iRev.gradient(x2, dy1, dy2);
                    String prefix = iRev.getPrefix();
                    dy1 = gradients.get(0);
                    dy2 = gradients.get(1);
                    grads[(cnt+1) * 3 - 3] = new Pair(prefix + "btnk_conv1Weight", gradients.get(2));
                    grads[(cnt+1) * 3 - 2] = new Pair(prefix + "btnk_conv2Weight", gradients.get(3));
                    grads[(cnt+1) * 3 - 1] = new Pair(prefix + "btnk_conv3Weight", gradients.get(4));
                    cnt -= 1;
                }
            }
            for (Pair<String, INDArray> grad : grads) {
                gradient.setGradientFor(grad.first, grad.second);
            }
        }
    }
}