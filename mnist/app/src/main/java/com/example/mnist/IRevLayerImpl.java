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
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

public class IRevLayerImpl extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected boolean first;
    protected long pad;
    protected int stride;
    protected InjectivePad InjPad;
    protected Psi psi;



    public IRevLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
        first = ((IRevLayer) conf().getLayer()).getfirst();
        pad = ((IRevLayer) conf().getLayer()).getNOut() * 2 - ((IRevLayer) conf().getLayer()).getNIn();
        stride = ((IRevLayer) conf().getLayer()).getStride();
        InjPad = new InjectivePad(pad, conf);
        psi = new Psi(conf, stride);
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




    }


    @Override
    public INDArray activate(INDArray[] x) {
        if (this.pad != 0 && this.stride == 1) {
            // append x

            input = input.permute(0, 2, 1, 3);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new ZeroPaddingLayer.Builder(0, (int)this.pad, 0, 0)
                            .build())
                    .setInputType(InputType.convolutionalFlat(input.shape()[0],input.shape()[1], input.shape()[2])) // InputType.convolutional for normal image
                    .build();
            MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
            myNetwork.init();
            INDArray output = myNetwork.output(input);
            output = output.permute(0, 2, 1, 3);

        }
        INDArray x1 = x[0];
        INDArray x2 = x[1];

    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}

