package com.example.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import android.util.Log;
import android.util.Pair;

import java.util.ArrayList;
import java.util.List;


public class IRevBlock {
    private int stride;
    private int pad;
    private String[] output;
    public Bottleneck bottleneck;


    public IRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                     int in_ch, int out_ch, int stride, int mult, String input1,
                     String input2, String prefix) {
//       input1: (3, 1, 32, 32)
//       input2: (3, 2, 32, 32)
        // in_ch: 3, out_ch: 16
        int pad = 2 * out_ch - in_ch; // 29
        int n = in_ch / 2;
        if (pad != 0 && stride == 1) {
            in_ch = out_ch * 2;
        }
        //input1: (3, 12, 16, 16) input2: (3, 12, 16, 16)
        this.pad = pad;
        this.stride = stride;
        this.bottleneck = new Bottleneck(in_ch/2, out_ch, stride,
                mult, WeightInit.XAVIER); // (3, 16, 32, 32)
        if (stride == 1 && pad != 0) {
            graphBuilder.addVertex(prefix + "merge", new MergeVertex(), input1, input2) //(3, 3, 32, 32)
                        //injective padding
                        .addLayer(prefix + "permute1", new PermuteLayer(0, 2, 3, 1), prefix + "merge")
                        .addLayer(prefix + "zeroPadding", new ZeroPaddingLayer(0, 0, 0, pad),prefix + "permute1")
                        .addLayer(prefix + "permute2", new PermuteLayer(0, 3, 1, 2), prefix + "zeroPadding") //(3, 32, 32, 32)
                        //finish injective padding
                        .addVertex(prefix + "x1", new SubsetVertexN(0, out_ch - 1), prefix + "permute2") // (3, 16, 32, 32)
                        .addVertex(prefix + "x2", new SubsetVertexN(out_ch, 2 * out_ch - 1), prefix + "permute2"); // (3, 16, 32, 32)
            input1 = prefix + "x1";
            input2 = prefix + "x2";
        }
        graphBuilder.addLayer(prefix + "btnk", this.bottleneck, input2);
        if (stride == 2) {
            Log.d("stride", " 2");
            ConvolutionLayer psi = new PsiLayer.Builder()
                    .BlockSize(stride)
                    .nIn(in_ch)
                    .nOut(out_ch)
                    .build();
            graphBuilder
                    .addLayer(prefix + "_psi1", psi, input1)
                    .addLayer(prefix + "_psi2", psi, input2);
            input1 = prefix + "_psi1";
            input2 = prefix + "_psi2";
            Log.d("stride", " 2 good");
        }

        graphBuilder
                .addVertex(prefix + "_y1", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix + "btnk", input1);
        this.output = new String[2];
        this.output[0] = input2;
        this.output[1] = prefix + "_y1";
    }

    public INDArray[] inverse(INDArray y1, INDArray y2) {

        if (this.stride == 1 && this.pad != 0) {
            //compute injective padding's inverse
            INDArray merge = Nd4j.concat(1, output0, output1);
            merge = merge.permute(1, 0, 2, 3);
            INDArray beforePadding =  merge.get(NDArrayIndex.interval(this.pad, merge.shape()[1] - this.pad));    // Here I think it is 2*pad, see if it is true.
            beforePadding = beforePadding.permute(1, 0, 2, 3);
            this.bottleneck.gradient()

        }

        if (this.stride == 2) {
            //call PSI inverse
            INDArray beforepsi = PsiLayerImpl.inverse(output0, stride);
        }

        return
    }

    // This function computes the total gradient of the graph without referring to the stored activation
    protected List<INDArray> gradient(INDArray x1, INDArray dy1, INDArray dy2) {
        INDArray dx1 = null;
        INDArray dx2 = null;
        INDArray dc1 = null;
        INDArray dc2 = null;
        INDArray dc3 = null;

        // TODO: downsample when switching btwn stages? Seems like we could ignore this since iRevNets
        // calculate the gradient w.r.t x1, x2 and list of weights
        // dz1 = S-1(dy1) + (dF_dz1).T.dot(dy2)
        // dx2 = S-1(dy2)
        // dx1 = dz1s
        if (this.stride == 1 && this.pad != 0) {
            INDArray z1 = x1;
            INDArray[] ds = this.bottleneck.gradient(z1, dy2); //dy2_x2, dc1, dc2, dx3
            INDArray dy2_x2 = ds[0];
            dc1 = ds[1];
            dc2 = ds[2];
            dc3 = ds[3];
            INDArray dy2_x1 = dy1;
            INDArray inverse_dx2 = dy2_x2.add(dy2_x1);
            INDArray inverse_dx1 = dy2;
            INDArray[] dx = injInverse(inverse_dx1, inverse_dx2);
            dx1 = dx[0];
            dx2 = dx[1];
        } else if (this.stride == 1 && this.pad == 0) {
            INDArray z1 = x1;
            INDArray[] ds = this.bottleneck.gradient(z1, dy2); //dy2_x2, dc1, dc2, dx3
            INDArray dy2_x2 = ds[0];
            dc1 = ds[1];
            dc2 = ds[2];
            dc3 = ds[3];
            INDArray dy2_x1 = dy1;
            dx2 = dy2_x2.add(dy2_x1);
            dx1 = dy2;
        } else if (this.stride == 2) {
            INDArray z1 = x1;
            INDArray[] ds = this.bottleneck.gradient(z1, dy2); //dy2_x2, dc1, dc2, dx3
            INDArray dy2_x2 = ds[0];
            dc1 = ds[1];
            dc2 = ds[2];
            dc3 = ds[3];
            INDArray dy2_x1 = PsiLayerImpl.inverse(dy1, this.stride);
            dx2 = dy2_x2.add(dy2_x1);
            dx1 = PsiLayerImpl.inverse(dy2, this.stride);
        }

        // return (dx1, dx2, dc1, dc2, dc3)
        List<INDArray> gradients = new ArrayList<INDArray>();
        gradients.add(dx1);
        gradients.add(dx2);
        gradients.add(dc1);
        gradients.add(dc2);
        gradients.add(dc3);
        return gradients;
    }


    protected String[] getOutput() {
        return this.output;
    }
}