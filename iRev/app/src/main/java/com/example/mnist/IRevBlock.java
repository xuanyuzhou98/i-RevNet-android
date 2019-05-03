package com.example.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import android.util.Log;



public class IRevBlock {
    private int stride;
    private String[] output;
    public Bottleneck bottleneck;


    public IRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                     int in_ch, int out_ch, int stride, int mult, String input1,
                     String input2, String prefix) {
        int pad = 2 * out_ch - in_ch;
        int n = in_ch / 2;
        if (pad != 0 && stride == 1) {
            in_ch = out_ch * 2;
        }
        //input1: (3, 12, 16, 16) input2: (3, 12, 16, 16)
        this.bottleneck = new Bottleneck(in_ch/2, out_ch, stride,
                mult, WeightInit.XAVIER);
        if (stride == 1 && pad != 0) {
            Log.d("stride", " 1");
            graphBuilder.addVertex(prefix + "merge", new MergeVertex(), input1, input2) //(3, 24, 16, 16)
                        //injective padding
                        .addLayer(prefix + "permute1", new PermuteLayer(0, 2, 1, 3), prefix + "merge") // (3, 16, 24, 16)
                        .addLayer(prefix + "zeroPadding", new ZeroPaddingLayer(0, 0, 0, pad),prefix + "permute1")
                        .addLayer(prefix + "permute2", new PermuteLayer(0, 2, 1, 3), prefix + "zeroPadding") //(3, 24, 16, 36)
                        //finish injective padding
                        .addVertex(prefix + "x1", new SubsetVertex(n-1, 0), prefix + "permute2") // (3, 12, 16, 36)
                        .addVertex(prefix + "x2", new SubsetVertex(in_ch-1, n), prefix + "permute2"); // (3, 12, 16, 36)
            Log.d("stride", " 1 good");
            input1 = prefix + "x1";
            input2 = prefix + "x2";;
        }
        graphBuilder.addLayer(prefix + "btnk", this.bottleneck, input2);
        if (stride == 2) {
            Log.d("stride", " 2");
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
            Log.d("stride", " 2 good");
        }
        // input1: (3, 12, 16, 36)
        // btnk: (3, 16,
        graphBuilder
                .addVertex(prefix + "_y1", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix + "btnk", input1);
        this.output = new String[2];
        this.output[0] = input2;
        this.output[1] = prefix + "_y1";
    }

    protected String[] getOutput() {
        return this.output;
    }
}