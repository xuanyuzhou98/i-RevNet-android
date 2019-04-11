package com.example.mnist;

import java.io.File;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

import android.os.Environment;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class MainActivity extends AppCompatActivity {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/mnist";
    private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute("");
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });
    }

    private class AsyncTaskRunner extends AsyncTask<String, Integer, String> {


        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
        }

        // This is our main background thread for the neural net
        @Override
        protected String doInBackground(String... params) {
            //number of rows and columns in the input pictures
            final int numRows = 28;
            final int numColumns = 28;
            int channels = 1; // single channel for grayscale images
            int outputNum = 10; // number of output classes
            int batchSize = 54; // batch size for each epoch
            int rngSeed = 1234; // random number seed for reproducibility
            int numEpochs = 1; // number of epochs to perform
            Random randNumGen = new Random(rngSeed);

            //Get the DataSetIterators:
            Log.d("load data", "Data load and vectorization...");


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

                ConvolutionLayer conv1 = new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build();
                int[] conv1OutShape = getConvLayerOutShape(numRows, numColumns, 5, 1);
                int conv1OutChannels = 20;
                int flopConv1 = getFlopCountConv(channels, 5, conv1OutChannels, conv1OutShape[0], conv1OutShape[1]);
                int flopConv1Back = getFlopCountConvBackward(channels, 5, conv1OutChannels, conv1OutShape[0], conv1OutShape[1]);

                SubsamplingLayer maxpool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build();
                int[] maxpool1OutShape = getConvLayerOutShape(conv1OutShape[0], conv1OutShape[1], 2, 2);
                int maxpool1OutChannels = conv1OutChannels;

                ConvolutionLayer conv2 = new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build();
                int[] conv2OutShape = getConvLayerOutShape(maxpool1OutShape[0], maxpool1OutShape[1], 5, 1);
                int conv2OutChannels = 50;
                int flopConv2 = getFlopCountConv(maxpool1OutChannels, 5, conv2OutChannels, conv2OutShape[0], conv2OutShape[1]);
                int flopConv2Back = getFlopCountConvBackward(channels, 5, conv1OutChannels, conv1OutShape[0], conv1OutShape[1]);

                SubsamplingLayer maxpool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build();
                int[] maxpool2OutShape = getConvLayerOutShape(conv2OutShape[0], conv2OutShape[1], 2, 2);
                int maxpool2OutChannels = conv2OutChannels;

                DenseLayer fc = new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build();
                int fcInShape = maxpool2OutShape[0] * maxpool2OutShape[1] * maxpool2OutChannels;
                int flopFC = getFlopCountFC(fcInShape, 500);
                int flopFCBack = getFlopCountFCBackward(fcInShape, 500);

                OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build();
                int totalFlopsForward = batchSize * (flopConv1 + flopConv2 + flopFC);
                int totalFlopsBackward = batchSize * (flopConv1Back + flopConv2Back + flopFCBack);
                Log.d("Flop count", "batch size " + batchSize);
                Log.d("Flop count", "forward count " + totalFlopsForward);
                Log.d("Flop count", "bacward count " + totalFlopsBackward);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(rngSeed)
                        .l2(0.0005) // ridge regression value
                        .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .setInputType(InputType.convolutionalFlat(numRows, numColumns, channels)) // InputType.convolutional for normal image
                        .layer(conv1)
                        .layer(maxpool1)
                        .layer(conv2)
                        .layer(maxpool2)
                        .layer(fc)
                        .layer(outputLayer)
                        .build();

                MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
                myNetwork.init();
                myNetwork.setListeners(new ScoreIterationListener(10));
                Log.d("Total num of params", "Total num of params" + myNetwork.numParams());

                Log.d("train model", "Train model....");
                for(int l=0; l<=numEpochs; l++) {
                    myNetwork.fit(mnistTrain);
                }

                Log.d("evaluate model", "Evaluate model....");
                Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
                while(mnistTest.hasNext()){
                    DataSet next = mnistTest.next();
                    INDArray output = myNetwork.output(next.getFeatures()); //get the networks prediction
                    eval.eval(next.getLabels(), output); //check the prediction against the true class
                }


                //Since we used global variables to store the classification results, no need to return
                //a results string. If the results were returned here they would be passed to onPostExecute.
                Log.d("evaluate stats", eval.stats());
                Log.d("finished","****************Example finished********************");
            }catch(Exception e){
                e.printStackTrace();
            }

            return "";
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
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
        }
    }

    private int[] getConvLayerOutShape(int inputH, int inputW, int filterSize, int stride) {
        int outputH = (inputH - filterSize) / stride + 1;
        int outputW = (inputW - filterSize) / stride + 1;
        return new int[] {outputH, outputW};
    }

    private int getFlopCountConv(int channels, int filter_size, int num_filters,
                                 int outShapeH, int outShapeW) {
        return (1 + 2 * channels * filter_size * filter_size) * num_filters * outShapeH * outShapeW;
    }

    private int getFlopCountFC(int inputSize, int outputSize) {
        return  (2 * inputSize - 1) * outputSize;
    }

    private int getFlopCountConvBackward(int channels, int filter_size, int num_filters,
                                         int outShapeH, int outShapeW) {
        int out = outShapeH * outShapeW;
        int db = out;
        int dw = num_filters *
                ((2 * out - 1) * channels * filter_size * filter_size);
        int dx_cols = channels * filter_size * filter_size * (2 * num_filters - 1) * out;
        int dx = channels * filter_size * filter_size * out;
        return db + dw + dx_cols + dx;
    }

    private int getFlopCountFCBackward(int inputSize, int outputSize) {
        int db = outputSize;
        int dx = (2 * outputSize - 1) * inputSize;
        int dw = inputSize * outputSize;
        return db + dx + dw;
    }

//    private int getFlopCountSigmoid(int outputSize) {
//        int forward = 4 * outputSize;
//        int backward = 0;
//        return  forward + backward;
//    }


}