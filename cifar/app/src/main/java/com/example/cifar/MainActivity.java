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
            // we will create configuration with 10MB memory space preallocated
            WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder()
                    .initialSize(10 * 1024L * 1024L)
                    .policyAllocation(AllocationPolicy.STRICT)
                    .policyLearning(LearningPolicy.NONE)
                    .build();

            try{
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
                cifarTrain.setPreProcessor(scaler);

                // vectorization of test data
                File testData = new File(basePath + "/cifar/test");
                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
                testRR.initialize(testSplit);
                DataSetIterator cifarTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
                cifarTest.setPreProcessor(scaler); // same normalization for better results


                NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
                        //.dataType(DataType.HALF)
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
                ActivationLayer relu = new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build();
                graph.addLayer("firstRelu", relu, "tilde_x0");
                int in_ch_Block = in_ch;
                int inputH = numRows;
                int inputW = numColumns;
                String input1 = "x0"; //(3, 1, 32, 32)
                String input2 = "firstRelu"; // (3, 2, 32, 32)
                List<IRevBlock> blockList = new ArrayList<>();
                for (int i = 0; i < 3; i++) { // for each stage
                    for (int j = 0; j < nBlocks[i]; j++) { // for each block in the stage
                        int stride = 1;
                        if (j == 0) {
                            stride = nStrides[i];
                        }
                        IRevBlock innerIRevBlock = new IRevBlock(graph, batchSize, inputH, inputW,
                                in_ch_Block, nChannels[i], stride,
                                mult, input1, input2, String.valueOf(i) + String.valueOf(j));
                        String[] outputs = innerIRevBlock.getOutput();
                        input1 = outputs[0];
                        input2 = outputs[1];
                        blockList.add(innerIRevBlock);
                        in_ch_Block = 2 * nChannels[i];  // input1 channel + input2 channel
                        inputH = innerIRevBlock.getOutputH();
                        inputW = innerIRevBlock.getOutputW();
                    }
                }

                ProbLayer probLayer = new ProbLayer(batchSize, inputH, inputW, nChannels[nChannels.length - 1] * 2, outputNum, 8, 8,
                        WeightInit.XAVIER);

//                graph.addVertex("merge", new MergeVertex(), input1, input2)
//                        .addLayer("outputProb", probLayer,"merge")
//                        .setOutputs("outputProb", "merge");
                graph.addVertex("merge", new MergeVertex(), input1, input2)
                        .addLayer("outputProb", probLayer,"merge")
                        .setOutputs("outputProb", "merge", "firstRelu",
                                "217btnk", "216btnk", "215btnk", "214btnk", "213btnk", "212btnk",
                                "211btnk", "210btnk", "29btnk", "28btnk", "27btnk", "26btnk",
                                "25btnk", "24btnk", "23btnk", "22btnk", "21btnk", "20btnk",
                                "117btnk", "116btnk", "115btnk", "114btnk", "113btnk", "112btnk",
                                "111btnk", "110btnk", "19btnk", "18btnk", "17btnk", "16btnk",
                                "15btnk", "14btnk", "13btnk", "12btnk", "11btnk", "10btnk",
                                "017btnk", "016btnk", "015btnk", "014btnk", "013btnk", "012btnk",
                                "011btnk", "010btnk", "09btnk", "08btnk", "07btnk", "06btnk",
                                "05btnk", "04btnk", "03btnk", "02btnk", "01btnk", "00btnk");

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
                            INDArray[] list = model.output(false, false, features);
                            INDArray merge = list[0];
                            Log.d("test", String.valueOf(list.length));
                            for(int k = 0; k < list.length; k++) {
                                Log.d("btnk", k + String.valueOf(list[k]));
                            }
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
            return "";
        }


        // This function computes the total gradient of the graph without referring to the stored activation
        protected HashMap<String, INDArray> computeGradient(ComputationGraph model, INDArray y2, INDArray y1, int[] nBlocks,
                                                            List<IRevBlock> blockList, INDArray[] lossGradient) {

            HashMap<String, INDArray> gradsResult = new HashMap<>();
            // get dy1 and dy2
            INDArray dy1 = lossGradient[0];
            INDArray dy2 = lossGradient[1];

            int cnt = blockList.size() - 1;
            // from the last layer to the first layer
            for (int i = nBlocks.length - 1; i >= 0; i -= 1) { // for each stage
                for (int j = nBlocks[i] - 1; j >= 0; j -= 1) { // for each iRevBlock
                    IRevBlock iRev = blockList.get(cnt);
                    // get x1 and x2
                    INDArray[] x = iRev.inverse(y1, y2);
                    INDArray x1 = x[0];
                    INDArray x2 = x[1];
                    // update (and swap) y1 and y2
                    y1 = x1;
                    y2 = x2;
                    // get gradients
                    List<INDArray> gradients = iRev.gradient(x1, dy1, dy2);
                    // update dy1 and dy2 (already swapped)
                    String prefix = iRev.getPrefix();
                    dy1 = gradients.get(0);
                    dy2 = gradients.get(1);

                    // save graidents
                    gradsResult.put(prefix + "btnk_conv1Weight", gradients.get(2));
                    gradsResult.put(prefix + "btnk_conv2Weight", gradients.get(3));
                    gradsResult.put(prefix + "btnk_conv3Weight", gradients.get(4));
                    cnt -= 1;
                }
            }

            return gradsResult;
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
}