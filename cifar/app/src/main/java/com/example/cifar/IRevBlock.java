package com.example.cifar;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;


public class IRevBlock {
    private String prefix;
    private int stride;
    private int pad;
    private String[] output;
    private Bottleneck bottleneck;

    public IRevBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder,
                     int in_ch, int out_ch, int stride, boolean first, int mult, String input1,
                     String input2, String prefix) {
        int pad = 2 * out_ch - in_ch;
        if (pad != 0 && stride == 1) {
            in_ch = out_ch * 2;
        }
        this.pad = pad;
        this.stride = stride;
        this.prefix = prefix;
        this.bottleneck = new Bottleneck(in_ch/2, out_ch, stride,
                mult, first);
        if (stride == 1 && pad != 0) {
            graphBuilder.addVertex(prefix + "merge", new MergeVertex(), input1, input2)
                        .addLayer(prefix + "permute1", new PermuteLayer(0, 2, 1, 3), prefix + "merge") //injective padding
                        .addLayer(prefix + "zeroPadding", new ZeroPaddingLayer(0, pad, 0, 0),prefix + "permute1")
                        .addLayer(prefix + "permute2", new PermuteLayer(0, 2, 1, 3), prefix + "zeroPadding") //finish injective padding
                        .addVertex(prefix + "x1", new SubsetVertexN(0, out_ch - 1), prefix + "permute2")
                        .addVertex(prefix + "x2", new SubsetVertexN(out_ch, 2 * out_ch - 1), prefix + "permute2");
            input1 = prefix + "x1";
            input2 = prefix + "x2";
        }
        graphBuilder.addLayer(prefix + "btnk", this.bottleneck, input2);
        if (stride == 2) {
            PsiLayer psi = new PsiLayer.Builder()
                    .BlockSize(stride)
                    .build();
            graphBuilder
                    .addLayer(prefix + "_psi1", psi, input1)
                    .addLayer(prefix + "_psi2", psi, input2);
            input1 = prefix + "_psi1";
            input2 = prefix + "_psi2";
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
        INDArray beforePadding =  merge.get(NDArrayIndex.interval(0, merge.shape()[0] - this.pad));
        long first = beforePadding.shape()[0] / 2;
        x[0] = beforePadding.get(NDArrayIndex.interval(0, first)).permute(1, 0, 2, 3);
        x[1] = beforePadding.get(NDArrayIndex.interval(first, beforePadding.shape()[0])).permute(1, 0, 2, 3);
        return x;
    }

    public INDArray[] inverse(INDArray y1, INDArray y2) {
        INDArray x1;
        INDArray x2;
        INDArray[] x = new INDArray[2];
        if (this.stride == 1 && this.pad != 0) {
            INDArray btnk = this.bottleneck.forward(y1);
            x1 = y2.sub(btnk);
            x2 = y1;
        }
        else if (this.stride == 1 && this.pad == 0) {
            x2 = y1;
            INDArray btnk = this.bottleneck.forward(x2);
            x1 = y2.sub(btnk);
        }
        else {
            x2 = PsiLayerImpl.inverse(y1, this.stride);
            INDArray btnk = this.bottleneck.forward(x2);
            INDArray px1 = y2.sub(btnk);
            x1 = PsiLayerImpl.inverse(px1, this.stride);
        }
        x[0] = x1;
        x[1] = x2;
        return x;
    }

    protected List<INDArray> gradient(INDArray x2, INDArray dy1, INDArray dy2) {
        INDArray dx1 = null;
        INDArray dx2 = null;
        INDArray dc1 = null;
        INDArray dc2 = null;
        INDArray dc3 = null;
        INDArray[] ds = null;
        if (this.stride == 1 && this.pad != 0) {
            INDArray dy1_injx2 = dy1;
            ds = this.bottleneck.gradient(x2, dy2);
            INDArray dy2_injx2 = ds[0];
            INDArray dinjx1 = dy2;
            INDArray dinjx2 = dy1_injx2.add(dy2_injx2);
            INDArray[] dx = injInverse(dinjx1, dinjx2);
            dx1 = dx[0];
            dx2 = dx[1];
        } else if (this.stride == 1 && this.pad == 0) {
            dx1 = dy2;
            INDArray dy1_x2 = dy1;
            ds = this.bottleneck.gradient(x2, dy2);
            INDArray dy2_x2 = ds[0];
            dx2 = dy2_x2.add(dy1_x2);
        } else if (this.stride == 2) {
            dx1 = PsiLayerImpl.inverse(dy2, this.stride);
            INDArray dy1_x2 = PsiLayerImpl.inverse(dy1, this.stride);
            ds = this.bottleneck.gradient(x2, dy2);
            INDArray dy2_x2 = ds[0];
            dx2 = dy1_x2.add(dy2_x2);
        }
        dc1 = ds[1];
        dc2 = ds[2];
        dc3 = ds[3];
        List<INDArray> gradients = new ArrayList<>();
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