package com.example.mnist;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.learning.config.Sgd;
import java.lang.Math;


public class MainActivity extends AppCompatActivity {

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
            int channels = 3;
            int init_ds = 2;
            int in_ch = channels * (int)Math.pow(2, init_ds);
            int n = in_ch / 2;
            int outputNum = 10; // number of output classes
            boolean first = true;
            int mult = 4;
            INDArray[] TestArray = new INDArray[1];
            INDArray sample = Nd4j.ones(1, 3, 28, 28);
            TestArray[0] = sample;
            ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                    .seed(1234)
                    .activation(Activation.IDENTITY)
                    .updater(new Sgd(0.05))
                    .weightInit(WeightInit.XAVIER)
                    .l1(1e-7)
                    .l2(5e-5)
                    .graphBuilder();
            graph.addInputs("input").setInputTypes(InputType.convolutionalFlat(28, 28, 3))
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
            graph.addVertex("merge", new MergeVertex(), output[0], output[1])
                    .addLayer("outputBN", new BatchNormalization.Builder()
                            .nIn(n * 4 * 2)
                            .nOut(n * 4 * 2)
                            .build(), "merge")
                    .addLayer("outputRelu", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                            "outputBN")
                    .addLayer("outputPool", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(),
                            "outputRelu")
                    .addLayer("output", new DenseLayer.Builder().activation(Activation.RELU)
                            .nOut(outputNum).build(), "outputPool")
                    .setOutputs("output");

            ComputationGraphConfiguration conf = graph.build();
            ComputationGraph model = new ComputationGraph(conf);
            model.init();
            INDArray testOutput = model.output(TestArray)[0];

            Log.d("Output", testOutput.toString());
            return "";
        }


        protected String[] iRevInverse(ComputationGraphConfiguration.GraphBuilder graphBuilder, INDArray x,
                                     int in_ch, int out_ch, int stride, boolean first, float dropout_rate,
                                     int mult, String input1, String input2, String prefix) {
//            x2, y1 = x[0], x[1]
////            if self.stride == 2:
////            x2 = self.psi.inverse(x2)
////            Fx2 = - self.bottleneck_block(x2)
////            x1 = Fx2 + y1
////            if self.stride == 2:
////            x1 = self.psi.inverse(x1)
////            if self.pad != 0 and self.stride == 1:
////            x = merge(x1, x2)
////            x = self.inj_pad.inverse(x)
////            x1, x2 = split(x)
////            x = (x1, x2)
////        else:
////            x = (x1, x2)
////            return x

            // Here I assume in_ch is the out_ch of forward function, we can modify it if we don't want in_ch in this function
            int n = in_ch / 2;


            graphBuilder
                    .addVertex(prefix + "_x2", new SubsetVertex(0, n-1), input1)
                    .addVertex(prefix + "_y1", new SubsetVertex(n, in_ch-1), input2);
            if (stride == 2) {
                graphBuilder
                        .addLayer(prefix+"_inverse_x2", new PsiLayer.Builder()
                                .BlockSize()
                        .build())
            }






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