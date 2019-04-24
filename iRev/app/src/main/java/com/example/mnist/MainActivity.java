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
                    .setOutputs("output", "outputPool");

            ComputationGraphConfiguration conf = graph.build();
            ComputationGraph model = new ComputationGraph(conf);
            model.init();
            INDArray testOutput = model.output(TestArray)[0];

            Log.d("Output", testOutput.toString());

//            int fcInShape = maxpool2OutShape[0] * maxpool2OutShape[1] * maxpool2OutChannels;
//            int flopFC = getFlopCountFC(fcInShape, outputNum);
//            int flopFCBack = getFlopCountFCBackward(fcInShape, outputNum);


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

            int batchSize = 54;
            int H = 28;
            int W = 28;

            int[] conv1OutShape = getConvLayerOutShape(H, W, 3, stride);
            int conv1OutChannels = out_ch/mult;
            int conv1InChannels = in_ch;
            int flopConv1 = getFlopCountConv(conv1InChannels, 3, conv1OutChannels, conv1OutShape[0], conv1OutShape[1]);
            int flopConv1Back = getFlopCountConvBackward(conv1InChannels, 3, conv1OutChannels, conv1OutShape[0], conv1OutShape[1]);

//            int[] bn1OutShape = getConvLayerOutShape(conv1OutShape[0], conv1OutShape[1], 2, 2);
//            int bn1OutChannels = conv1OutChannels;

            int[] conv2OutShape = getConvLayerOutShape(H, W, 3, stride);
            int conv2OutChannels = out_ch/mult;
            int conv2InChannels = out_ch/mult;
            int flopConv2 = getFlopCountConv(conv2InChannels, 3, conv2OutChannels, conv2OutShape[0], conv2OutShape[1]);
            int flopConv2Back = getFlopCountConvBackward(conv2InChannels, 3, conv2OutChannels, conv2OutShape[0], conv2OutShape[1]);

            int[] conv3OutShape = getConvLayerOutShape(H, W, 3, stride);
            int conv3OutChannels = out_ch;
            int conv3InChannels = out_ch/mult;
            int flopConv3 = getFlopCountConv(conv3InChannels, 3, conv3OutChannels, conv3OutShape[0], conv3OutShape[1]);
            int flopConv3Back = getFlopCountConvBackward(conv3InChannels, 3, conv3OutChannels, conv3OutShape[0], conv3OutShape[1]);

            int totalFlopsForward = batchSize * (flopConv1 + flopConv2 + flopConv3);
            int totalFlopsBackward = batchSize * (flopConv1Back + flopConv2Back + flopConv3Back);

            Log.d("BottleNeck Flop count", "batch size " + batchSize);
            Log.d("BottleNeck Flop count", "forward count " + totalFlopsForward);
            Log.d("BottleNeck Flop count", "bacward count " + totalFlopsBackward);

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

        private int[] getConvLayerOutShape(int inputH, int inputW, int filterSize, int stride) {
            int outputH = (inputH - filterSize) / stride + 1;
            int outputW = (inputW - filterSize) / stride + 1;
            return new int[] {outputH, outputW};
        }

        private int getFlopCountConv(int channels, int filter_size, int num_filters,
                                     int outShapeH, int outShapeW) {
            return (2 * channels * filter_size * filter_size - 1) * num_filters * outShapeH * outShapeW;
        }

        private int getFlopCountFC(int inputSize, int outputSize) {
            return  (2 * inputSize - 1) * outputSize;
        }

        private int getFlopCountConvBackward(int channels, int filter_size, int num_filters,
                                             int outShapeH, int outShapeW) {
            int out = outShapeH * outShapeW;
            int db = out;
            int dw = num_filters * ((2 * out - 1) * channels * filter_size * filter_size);
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
    }
}