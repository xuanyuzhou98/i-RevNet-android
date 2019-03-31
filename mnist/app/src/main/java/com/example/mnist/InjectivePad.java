package com.example.mnist;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;

public class InjectivePad extends BaseLayer<ConvolutionLayer> {
    protected int padSize;
    protected KerasZeroPadding2D pad;
    public InjectivePad(int PadSize, NeuralNetConfiguration conf){
        super(conf);
        padSize = PadSize;
        pad = new KerasZeroPadding2D("padding"=((0, padSize), (0, 0)))


    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

}
