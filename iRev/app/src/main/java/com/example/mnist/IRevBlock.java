package com.example.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;


public class IRevBlock {
    private int stride;
    private String[] output;
    public Bottleneck bottleneck;


    public IRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                     int in_ch, int out_ch, int stride, boolean first,
                     int mult, String input1, String input2, String prefix) {
        this.bottleneck = new Bottleneck(in_ch, out_ch, stride, first,
                mult, WeightInit.XAVIER);
        graphBuilder
                .addLayer("btnk", this.bottleneck, input2);
        if (stride == 2) {
            ConvolutionLayer psi = new PsiLayer.Builder()
                    .BlockSize(stride)
                    .nIn(in_ch)
                    .nOut(out_ch)
                    .build();
            this.stride = stride;
            graphBuilder
                    .addLayer(prefix + "_psi1", psi, input1)
                    .addLayer(prefix + "_psi2", psi, input2);
            input1 = prefix + "_psi1";
            input2 = prefix + "_psi2";
        }
        graphBuilder
                .addVertex(prefix + "_y1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "btnk", input1);
        this.output = new String[2];
        this.output[0] = input2;
        this.output[1] = prefix + "_y1";
        this.output = new String[1];
        this.output[0] = "btnk";

    }

    protected String[] getOutput() {
        return this.output;
    }
}