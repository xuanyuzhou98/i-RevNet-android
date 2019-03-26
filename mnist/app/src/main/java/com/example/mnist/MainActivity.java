package com.example.mnist;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.evaluation.classification.Evaluation;

public class MainActivity extends AppCompatActivity {

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
            int outputNum = 10; // number of output classes
            int batchSize = 64; // batch size for each epoch
            int rngSeed = 123; // random number seed for reproducibility
            int numEpochs = 15; // number of epochs to perform
            double rate = 0.0015; // learning rate

            //Get the DataSetIterators:
            Log.d("build model", "Build model....");
            try {
                DataSetIterator mnistTrain = new MnistDataSetIterator( batchSize, true, rngSeed);
                DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
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
            }catch(Exception e) {
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