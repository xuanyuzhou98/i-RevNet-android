package com.example.mnist;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
i

public class IRevLayerImpl extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected boolean first;
    protected long pad;
    protected int stride;


    public IRevLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
        first = ((IRevLayer) conf().getLayer()).getfirst();
        pad = ((IRevLayer) conf().getLayer()).getNOut() * 2 - ((IRevLayer) conf().getLayer()).getNIn();
        stride = ((IRevLayer) conf().getLayer()).getStride();
        this.pad = ((IRevLayer) conf().getLayer()).getPad();
        long in_ch = ((IRevLayer) conf().getLayer()).getNIn();
        long out_ch = ((IRevLayer) conf().getLayer()).getNOut();
        int mult = ((IRevLayer) conf().getLayer()).getMult();
        boolean affineBN = ((IRevLayer) conf().getLayer()).getAffineBN();
        double DropOutRate = ((IRevLayer) conf().getLayer()).getDropOutRate();
        if (pad != 0 && stride == 1){
            in_ch = out_ch * 2;
        }
        int rngSeed = 1234; // random number seed for reproducibility
        INDArray layers;
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .list()
                .layer(new BatchNormalization.Builder(in_ch / 2, affineBN)
                        .build())
                .layer( new ActivationLayer(Activation.RELU)) // Not sure
                .layer(new ConvolutionLayer.Builder(3)
                        .nIn(in_ch/2)
                        .stride(stride) // nIn need not specified in later layers
                        .nOut(out_ch/mult)
                        .build())
                .layer(new BatchNormalization.Builder(out_ch / mult, affineBN)
                        .build())
                .layer( new ActivationLayer(Activation.RELU)) // Not sure
                .layer(new ConvolutionLayer.Builder(3)
                        .nIn(out_ch/mult)
                        .stride(stride) // nIn need not specified in later layers
                        .nOut(out_ch/mult)
                        .build())
                .layer(new DropoutLayer.Builder(DropOutRate).build())

                .layer(new BatchNormalization.Builder(out_ch / mult, affineBN)
                        .build())
                .layer( new ActivationLayer(Activation.RELU)) // Not sure
                .layer(new ConvolutionLayer.Builder(3)
                        .nIn(out_ch/mult)
                        .stride(stride) // nIn need not specified in later layers
                        .nOut(out_ch)
                        .build())
                .build();
        MultiLayerNetwork myNetwork = new MultiLayerNetwork(config);
        myNetwork.init();
        myNetwork.output();




    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (this.pad != 0 && this.stride == 1) {
            // split along dim 0 and concat along dim 1
            INDArray[] x = Utils.split(input, 2, 0);
            INDArray temp = Nd4j.concat(1, x[0], x[1]);
            // injection padding
            temp = temp.permute(0, 2, 1, 3);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new ZeroPaddingLayer.Builder(0, (int)this.pad, 0, 0)
                            .build())
                    .setInputType(InputType.convolutionalFlat(temp.shape()[0],temp.shape()[1], temp.shape()[2])) // InputType.convolutional for normal image
                    .build();
            MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
            myNetwork.init();
            INDArray output = myNetwork.output(input);
            output = output.permute(0, 2, 1, 3);
            // split
            x = Utils.split(temp, 2, 1);
        }
        INDArray x1 = x[0];
        INDArray x2 = x[1];


    }

    public INDArray psi(INDArray input, long blockSize) {
        long blockSizeSq = blockSize * blockSize;
        INDArray output = input.permute(0, 2, 3, 1);
        long[] shape = output.shape();
        System.out.print(shape);
        assert shape.length >= 4;
        long batchSize = shape[0];
        long sHeight = shape[1];
        long sWidth = shape[2];
        long sDepth = shape[3];
        long dDepth = sDepth * blockSizeSq;
        long dHeight = sHeight / blockSize;

        for (int i = 0; i < sWidth; i += blockSize) {
            output = Nd4j.stack(2, output, output.slice(i, 2).reshape(batchSize, dHeight, dDepth));
        }

        output = output.permute(0, 2, 1, 3);
        output = output.permute(0, 3, 1, 2);

        return output;

    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}

