package com.example.mnist;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer;

public class IRevLayerImpl extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected boolean first;
    protected long pad;
    protected int stride;
    protected MultiLayerNetwork bottleneck;
    protected ZeroPaddingLayer zeropad;


    public IRevLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
        first = ((IRevLayer) conf().getLayer()).getfirst();
        pad = ((IRevLayer) conf().getLayer()).getNOut() * 2 - ((IRevLayer) conf().getLayer()).getNIn();
        stride = ((IRevLayer) conf().getLayer()).getStride();
        this.pad = 2 * ((IRevLayer) conf().getLayer()).getNOut() -
                   ((IRevLayer) conf().getLayer()).getNIn();
        long in_ch = ((IRevLayer) conf().getLayer()).getNIn();
        long out_ch = ((IRevLayer) conf().getLayer()).getNOut();
        int mult = ((IRevLayer) conf().getLayer()).getMult();
        boolean affineBN = ((IRevLayer) conf().getLayer()).getAffineBN();
        double DropOutRate = ((IRevLayer) conf().getLayer()).getDropOutRate();
        this.zeropad = new ZeroPaddingLayer(conf);
        if (pad != 0 && stride == 1){
            in_ch = out_ch * 2;
        }
        int rngSeed = 1234; // random number seed for reproducibility
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
        this.bottleneck = new MultiLayerNetwork(config);
        this.bottleneck.init();
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (this.pad != 0 && this.stride == 1) {
            INDArray[] inputlist = Utils.split(input, 2, 0);
            // split along dim 0 and concat along dim 1
            input = Nd4j.concat(1, inputlist[0], inputlist[1]);
            // injection padding
            input = input.permute(0, 2, 1, 3);
            input = this.zeropad.activate(training, workspaceMgr); // need to pass (0, 0, 0, pad_size)
            input = input.permute(0, 2, 1, 3);
            // split
            inputlist = Utils.split(input, 2, 1);
            input = Nd4j.concat(0, inputlist[0], inputlist[1]);
        }
        INDArray[] inputlist = Utils.split(input, 2, 0);
        INDArray x1 = inputlist[0];
        INDArray x2 = inputlist[1];
        INDArray Fx2 = this.bottleneck.output(x2);
        if (this.stride == 2) {
           x1 = psi(inputlist[0],this.stride);
           x2 = psi(inputlist[1],this.stride);
        }
        INDArray y1 = Fx2.add(x1);
        return  Nd4j.concat(0, x2, y1);
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

