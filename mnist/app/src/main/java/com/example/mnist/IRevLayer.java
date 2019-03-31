package com.example.mnist;
import org.deeplearning4j.nn.layers.BaseLayer;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;



public class IRevLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected ConvolutionMode convolutionMode;
    public IRevLayer(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
        convolutionMode = ((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf().getLayer()).getConvolutionMode();
    }



    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {

    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}

