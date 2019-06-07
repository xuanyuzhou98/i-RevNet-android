package com.example.cifar;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;


public class IRevBlock {
    private String prefix;
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
        this.prefix = prefix;
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
            PsiLayer psi = new PsiLayer.Builder()
                    .BlockSize(stride)
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


    public INDArray[] injInverse(INDArray input1, INDArray input2) {
        INDArray[] x = new INDArray[2];
        INDArray merge = Nd4j.concat(1, input1, input2);
        merge = merge.permute(1, 0, 2, 3);
        INDArray beforePadding =  merge.get(NDArrayIndex.interval(0, merge.shape()[0] - this.pad)); // exclusive last term
//            merge.get(NDArrayIndex.interval(0, merge.shape()[0] - this.pad),
//                    NDArrayIndex.interval(0, merge.shape()[1]), NDArrayIndex.interval(0, merge.shape()[2]),
//                    NDArrayIndex.interval(0, merge.shape()[3]));


        long first = beforePadding.shape()[0] / 2;
        x[0] = beforePadding.get(NDArrayIndex.interval(0, first));
        x[0] = x[0].permute(1, 0, 2, 3);
        x[1] = beforePadding.get(NDArrayIndex.interval(first, beforePadding.shape()[0]));
        x[1] = x[1].permute(1, 0, 2, 3);

        return x;
    }

    public INDArray[] inverse(INDArray y1, INDArray y2) {
        INDArray x1;
        INDArray x2;
        INDArray[] x = new INDArray[2];
        if (this.stride == 1 && this.pad != 0) {
            //compute injective padding's inverse
            INDArray btnk = this.bottleneck.forward(y1);
            INDArray input1 = y2.sub(btnk);
            INDArray input2 = y1;
            //x = injInverse(input1, input2);
            x1 = input1;
            x2 = input2;
//            INDArray merge = Nd4j.concat(1, input1, input2);
//            merge = merge.permute(1, 0, 2, 3);
//            INDArray beforePadding =  merge.get(NDArrayIndex.interval(0, merge.shape()[0] - this.pad)); // exclusive last term
////            merge.get(NDArrayIndex.interval(0, merge.shape()[0] - this.pad),
////                    NDArrayIndex.interval(0, merge.shape()[1]), NDArrayIndex.interval(0, merge.shape()[2]),
////                    NDArrayIndex.interval(0, merge.shape()[3]));
//
//
//            long first = beforePadding.shape()[0] / 2;
//            x1 = beforePadding.get(NDArrayIndex.interval(0, first));
//            x2 = beforePadding.get(NDArrayIndex.interval(first, beforePadding.shape()[0]));
//            //beforePadding = beforePadding.permute(1, 0, 2, 3);

        }
        else if (this.stride == 1 && this.pad == 0) {
            x2 = y1;
            INDArray btnk = this.bottleneck.forward(x2);
            x1 = y2.sub(btnk);
        }
        else {
            //call PSI inverse
            x2 = PsiLayerImpl.inverse(y1, this.stride);
            INDArray btnk = this.bottleneck.forward(x2);
            INDArray px1 = y2.sub(btnk);
            x1 = PsiLayerImpl.inverse(px1, this.stride);
        }

        // (this.stride == 2)

        x[0] = x1;
        x[1] = x2;

        return x;
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

    protected String getPrefix() {
        return this.prefix;
    }


    protected String[] getOutput() {
        return this.output;
    }
}