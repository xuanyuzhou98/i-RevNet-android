package com.example.cifar;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.RandomCropTransform;
import org.deeplearning4j.util.ModelSerializer;
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
import org.nd4j.linalg.dataset.api.preprocessor.StandardizeStrategy;
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/cifar";
    private static final String dataUrl = "http://pjreddie.com/media/files/cifar.tgz";
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
            UnitTestIrevBlockInverse();
//            try{
//                int[] nChannels = new int[]{16, 64, 256};
//                int[] nBlocks = new int[]{18, 18, 18};
//                int[] nStrides = new int[]{1, 2, 2};
//                int channels = 3;
//                int init_ds = 0;
//                int in_ch = channels * (int) Math.pow(2, init_ds);
//                int n = in_ch / 2;
//                int outputNum = 10; // number of output classes
//                final int numRows = 32;
//                final int numColumns = 32;
//                int rngSeed = 1234; // random number seed for reproducibility
//                int numEpochs = 1; // number of epochs to perform
//                Random randNumGen = new Random(rngSeed);
//                int batchSize = 16; // batch size for each epoch
//                int mult = 4;
//
//                if (!new File(basePath + "/cifar").exists()) {
//                    Log.d("Data download", "Data downloaded from " + dataUrl);
//                    String localFilePath = basePath + "/cifar.tgz";
//                    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
//                        DataUtilities.extractTarGz(localFilePath, basePath);
//                    }
//                }
//
//                // vectorization of train data
//                File trainData = new File(basePath + "/cifar/train");
//                FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
//                ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
//                trainRR.initialize(trainSplit);
//                DataSetIterator cifarTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
//                // pixel values from 0-255 to 0-1 (min-max scaling)
//                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//                scaler.fit(cifarTrain);
////                ImageTransform transform = new MultiImageTransform(
////                        new CropImageTransform(10),
////                        new FlipImageTransform(1));
//                cifarTrain.setPreProcessor(scaler);
//
//                // vectorization of test data
//                File testData = new File(basePath + "/cifar/test");
//                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//                ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
//                testRR.initialize(testSplit);
//                DataSetIterator cifarTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
//                cifarTest.setPreProcessor(scaler); // same normalization for better results
//
//
//                NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
//                        .seed(rngSeed)
//                        .activation(Activation.IDENTITY)
//                        .updater(new Nesterovs(0.1, 0.9))
//                        .weightInit(WeightInit.XAVIER)
//                        .l1(1e-7)
//                        .l2(5e-5);
//                if (half_precision) {
//                    config.dataType(DataType.HALF);
//                }
//                ComputationGraphConfiguration.GraphBuilder graph = config.graphBuilder();
//
//                graph.addInputs("input").setInputTypes(InputType.convolutional(numRows, numColumns, channels)); //(3, 3, 32, 32)
//                String lastLayer = "input";
//                if (init_ds != 0) {
//                    graph.addLayer("init_psi", new PsiLayer.Builder()
//                            .BlockSize(init_ds)
//                            .build(), "input");
//                    lastLayer = "init_psi";
//                }
//
//                graph.addVertex("x0", new SubsetVertexN(0, n - 1), lastLayer) //(3, 1, 32, 32)
//                        .addVertex("tilde_x0", new SubsetVertexN(n, in_ch - 1), lastLayer); //(3, 2, 32, 32)
//                int in_ch_Block = in_ch;
//                String input1 = "x0"; //(3, 1, 32, 32)
//                String input2 = "tilde_x0"; // (3, 2, 32, 32)
//                boolean first = true;
//                List<IRevBlock> blockList = new ArrayList<>();
//                for (int i = 0; i < 3; i++) { // for each stage
//                    for (int j = 0; j < nBlocks[i]; j++) { // for each block in the stage
//                        int stride = 1;
//                        if (j == 0) {
//                            stride = nStrides[i];
//                        }
//                        IRevBlock innerIRevBlock = new IRevBlock(graph, in_ch_Block, nChannels[i], stride, first,
//                                mult, input1, input2, String.valueOf(i) + String.valueOf(j));
//                        String[] outputs = innerIRevBlock.getOutput();
//                        input1 = outputs[0];
//                        input2 = outputs[1];
//                        blockList.add(innerIRevBlock);
//                        in_ch_Block = 2 * nChannels[i];
//                        first = false;
//                    }
//                }
//
//                ProbLayer probLayer = new ProbLayer(nChannels[nChannels.length - 1] * 2, outputNum, 8, 8,
//                        WeightInit.XAVIER);
//
//                OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(outputNum)
//                        .activation(Activation.SOFTMAX)
//                        .build();
//
//                graph.addVertex("merge", new MergeVertex(), input1, input2)
//                        .addLayer("outputProb", probLayer,"merge")
//                        .setOutputs("outputProb", "merge");
//
//                ComputationGraphConfiguration conf = graph.build();
//                ComputationGraph model = new ComputationGraph(conf);
//                model.init();
//                MemoryManager mg = Nd4j.getMemoryManager();
//                mg.togglePeriodicGc(true);
//                model.setListeners(new ScoreIterationListener(1));
//
//                Log.d("Output", "start training");
//                if (manual_gradients) {
//                    int i = 0;
//                    for (int epoch = 0; epoch <= numEpochs; epoch++) {
//                        Log.d("Epoch", "Running epoch " + epoch);
//                        while (cifarTrain.hasNext()) {
//                            Log.d("Iteration", "Running iter " + i);
//                            DataSet data = cifarTrain.next();
//                            INDArray label = data.getLabels();
//                            INDArray features = data.getFeatures();
//                            long StartTime = System.nanoTime();
//                            INDArray[] os =  model.output(false, false, features);
//                            INDArray merge = os[1];
//                            long EndTime = System.nanoTime();
//                            double elapsedTimeInSecond = (double) (EndTime - StartTime) / 1_000_000_000;
//                            Log.d("forward time", String.valueOf(elapsedTimeInSecond));
//                            Log.d("output", "finished forward iter " + i);
//
//                            StartTime = System.nanoTime();
//                            Gradient gradient = new DefaultGradient();
//                            INDArray[] outputGradients = probLayer.gradient(merge, label);
//                            INDArray dwGradient = outputGradients[1];
//                            INDArray dbGradient = outputGradients[2];
//                            gradient.setGradientFor("outputProb_denseWeight", dwGradient);
//                            gradient.setGradientFor("outputProb_denseBias", dbGradient);
//                            INDArray[] lossGradient = Utils.splitHalf(outputGradients[0]);
//                            INDArray[] hiddens = Utils.splitHalf(merge);
//                            HashMap<String, INDArray> gradientMap = computeGradient(model, hiddens[0], hiddens[1],
//                                    nBlocks, blockList, lossGradient);
//                            for (Map.Entry<String, INDArray> entry : gradientMap.entrySet()) {
//                                gradient.setGradientFor(entry.getKey(), entry.getValue());
//                            }
//                            model.getUpdater().update(gradient, i, epoch, batchSize, LayerWorkspaceMgr.noWorkspaces());
//                            EndTime = System.nanoTime();
//                            elapsedTimeInSecond = (double) (EndTime - StartTime) / 1_000_000_000;
//                            Log.d("backward time", String.valueOf(elapsedTimeInSecond));
//                            Log.d("output", "finished backward iter " + i);
//                            i++;
//                        }
//                    }
//                } else {
//                    for(int l=0; l <= numEpochs; l++) {
//                        model.fit(cifarTrain);
//                    }
//                }
//            } catch (Exception ex) {
//                ex.printStackTrace();
//            }
            return "";
        }

        private void UnitTestIrevBlockInverse() {
            try{
                int[] nChannels = new int[]{16, 64, 256};
                int[] nBlocks = new int[]{1, 0, 0};
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
                int batchSize = 16; // batch size for each epoch
                int mult = 4;

                if (!new File(basePath + "/cifar").exists()) {
                    Log.d("Data download", "Data downloaded from " + dataUrl);
                    String localFilePath = basePath + "/cifar.tgz";
                    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
                        DataUtilities.extractTarGz(localFilePath, basePath);
                    }
                }

                // vectorization of train data
                File trainData = new File(basePath + "/cifar/train");
                FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
                ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                trainRR.initialize(trainSplit);
                DataSetIterator cifarTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
                // pixel values from 0-255 to 0-1 (min-max scaling)
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.fit(cifarTrain);
//                ImageTransform transform = new MultiImageTransform(
//                        new CropImageTransform(10),
//                        new FlipImageTransform(1));
                cifarTrain.setPreProcessor(scaler);

                // vectorization of test data
                File testData = new File(basePath + "/cifar/test");
                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                testRR.initialize(testSplit);
                DataSetIterator cifarTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
                cifarTest.setPreProcessor(scaler); // same normalization for better results


                NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
                        .seed(rngSeed)
                        .activation(Activation.IDENTITY)
                        .updater(new Nesterovs(0.1, 0.9))
                        .weightInit(WeightInit.XAVIER)
                        .l1(1e-7)
                        .l2(5e-5);
                if (half_precision) {
                    config.dataType(DataType.HALF);
                }
                ComputationGraphConfiguration.GraphBuilder graph = config.graphBuilder();

                graph.addInputs("input").setInputTypes(InputType.convolutional(numRows, numColumns, channels)); //(3, 3, 32, 32)
                String lastLayer = "input";
                if (init_ds != 0) {
                    graph.addLayer("init_psi", new PsiLayer.Builder()
                            .BlockSize(init_ds)
                            .build(), "input");
                    lastLayer = "init_psi";
                }

                graph.addVertex("x0", new SubsetVertexN(0, n - 1), lastLayer) //(3, 1, 32, 32)
                        .addVertex("tilde_x0", new SubsetVertexN(n, in_ch - 1), lastLayer); //(3, 2, 32, 32)
                String input1 = "x0"; //(3, 1, 32, 32)
                String input2 = "tilde_x0"; // (3, 2, 32, 32)
                boolean first = true;
                List<IRevBlock> blockList = new ArrayList<>();

                IRevBlock innerIRevBlock = new IRevBlock(graph, in_ch, nChannels[0], nStrides[0], first,
                        mult, input1, input2, "00");
                String[] outputs = innerIRevBlock.getOutput();
                input1 = outputs[0];
                input2 = outputs[1];
                blockList.add(innerIRevBlock);

                ProbLayer probLayer = new ProbLayer(nChannels[0]*2, outputNum, 32, 32,
                        WeightInit.XAVIER);

                graph.addVertex("merge", new MergeVertex(), input1, input2)
                        .addLayer("outputProb", probLayer,"merge")
                        .setOutputs("outputProb", "merge", "x0", "tilde_x0");

                ComputationGraphConfiguration conf = graph.build();
                ComputationGraph model = new ComputationGraph(conf);
                model.init();
                MemoryManager mg = Nd4j.getMemoryManager();
                mg.togglePeriodicGc(true);
                model.setListeners(new ScoreIterationListener(1));

                Log.d("Output", "start training");
                if (manual_gradients) {
                    int i = 0;
                    for (int epoch = 0; epoch <= numEpochs; epoch++) {
                        Log.d("Epoch", "Running epoch " + epoch);
                        while (cifarTrain.hasNext()) {
                            Log.d("Iteration", "Running iter " + i);
                            DataSet data = cifarTrain.next();
                            INDArray label = data.getLabels();
                            INDArray features = data.getFeatures();
                            long StartTime = System.nanoTime();
                            INDArray[] os =  model.output(false, false, features);
                            INDArray merge = os[1];
                            long EndTime = System.nanoTime();
                            double elapsedTimeInSecond = (double) (EndTime - StartTime) / 1_000_000_000;
                            Log.d("forward time", String.valueOf(elapsedTimeInSecond));
                            Log.d("output", "finished forward iter " + i);

                            StartTime = System.nanoTime();
                            Gradient gradient = new DefaultGradient();
                            INDArray[] outputGradients = probLayer.gradient(merge, label);
                            INDArray dwGradient = outputGradients[1];
                            INDArray dbGradient = outputGradients[2];
                            gradient.setGradientFor("outputProb_denseWeight", dwGradient);
                            gradient.setGradientFor("outputProb_denseBias", dbGradient);
                            INDArray[] lossGradient = Utils.splitHalf(outputGradients[0]);
                            INDArray[] hiddens = Utils.splitHalf(merge);

                            INDArray x0 = os[2];
                            INDArray tilde_x0 = os[3];
                            INDArray[] inverseTest = blockList.get(0).inverse(hiddens[0], hiddens[1]);
                            Log.d("x0", x0.toString());
                            Log.d("x0Inverse", inverseTest[0].toString());
                            Log.d("tilde_x0", tilde_x0.toString());
                            Log.d("tilde_x0Inverse", inverseTest[1].toString());
                            

                            HashMap<String, INDArray> gradientMap = computeGradient(model, hiddens[0], hiddens[1],
                                    nBlocks, blockList, lossGradient);
                            for (Map.Entry<String, INDArray> entry : gradientMap.entrySet()) {
                                gradient.setGradientFor(entry.getKey(), entry.getValue());
                            }
                            model.getUpdater().update(gradient, i, epoch, batchSize, LayerWorkspaceMgr.noWorkspaces());
                            EndTime = System.nanoTime();
                            elapsedTimeInSecond = (double) (EndTime - StartTime) / 1_000_000_000;
                            Log.d("backward time", String.valueOf(elapsedTimeInSecond));
                            Log.d("output", "finished backward iter " + i);
                            i++;
                        }
                    }
                } else {
                    for(int l=0; l <= numEpochs; l++) {
                        model.fit(cifarTrain);
                    }
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }

        // This function computes the total gradient of the graph without referring to the stored activation
        protected HashMap<String, INDArray> computeGradient(ComputationGraph model, INDArray y1, INDArray y2, int[] nBlocks,
                                                            List<IRevBlock> blockList, INDArray[] lossGradient) {

            HashMap<String, INDArray> gradsResult = new HashMap<>();
            INDArray dy1 = lossGradient[0];
            INDArray dy2 = lossGradient[1];
            int cnt = blockList.size() - 1;
            // from the last layer to the first layer
            for (int i = nBlocks.length - 1; i >= 0; i -= 1) { // for each stage
                for (int j = nBlocks[i] - 1; j >= 0; j -= 1) { // for each iRevBlock
                    IRevBlock iRev = blockList.get(cnt);
                    INDArray[] x = iRev.inverse(y1, y2);
                    INDArray x1 = x[0];
                    INDArray x2 = x[1];
                    y1 = x1;
                    y2 = x2;
                    List<INDArray> gradients = iRev.gradient(x2, dy1, dy2);
                    String prefix = iRev.getPrefix();
                    dy1 = gradients.get(0);
                    dy2 = gradients.get(1);
                    gradsResult.put(prefix + "btnk_conv1Weight", gradients.get(2));
                    gradsResult.put(prefix + "btnk_conv2Weight", gradients.get(3));
                    gradsResult.put(prefix + "btnk_conv3Weight", gradients.get(4));
                    cnt -= 1;
                }
            }
            return gradsResult;
        }
    }
}