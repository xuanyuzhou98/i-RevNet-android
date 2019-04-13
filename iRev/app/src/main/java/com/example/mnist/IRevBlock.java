package com.example.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class IRevBlock {
    protected ComputationGraphConfiguration.GraphBuilder graph;

    protected  IRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        this.graph = graphBuilder;
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
        return prefix;
    }

    protected String[] iRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                                 int in_ch, int out_ch, int stride, boolean first, float dropout_rate,
                                 int mult, String input1, String input2, String prefix) {
        String Fx2 = bottleneckBlock(graphBuilder, in_ch, out_ch, stride, first, dropout_rate,
                mult, input2, prefix + "_btnk");
        if (stride == 2) {   // depth increasement
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
        this.graph.get
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

}
