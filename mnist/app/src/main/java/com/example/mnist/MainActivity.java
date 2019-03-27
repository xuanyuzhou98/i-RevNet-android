package com.example.mnist;

import java.io.File;
import java.util.Random;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

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

public class MainActivity extends AppCompatActivity {
    private static final String basePath = System.getProperty("java.io.tmpdir") + "/mnist";
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
            int batchSize = 64; // batch size for each epoch
            int rngSeed = 123; // random number seed for reproducibility
            int numEpochs = 15; // number of epochs to perform
            double rate = 0.0015; // learning rate
            Random randNumGen = new Random(rngSeed);

            //Get the DataSetIterators:
            Log.d("load data", "Data load and vectorization...");
            String localFilePath = basePath + "/mnist_png.tar.gz";
            try {
                if (DataUtilities.downloadFile(dataUrl, localFilePath))
                    Log.d("Data download", "Data downloaded from " + dataUrl);
                if (!new File(basePath + "/mnist_png").exists())
                    DataUtilities.extractTarGz(localFilePath, basePath);
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
                //build the layers of the network
                DenseLayer inputLayer = new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(500)
                        .build();

                //build the layers of the network
                DenseLayer hiddenLayer = new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(100)
                        .build();

                OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNum)
                        .build();

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(rngSeed) //include a random seed for reproducibility
                        // use stochastic gradient descent as an optimization algorithm
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(rate, 0.98))
                        .l2(rate * 0.005)
                        .list()
                        .layer(inputLayer)
                        .layer(hiddenLayer)
                        .layer(outputLayer)
                        .build();

                MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
                myNetwork.init();

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
}