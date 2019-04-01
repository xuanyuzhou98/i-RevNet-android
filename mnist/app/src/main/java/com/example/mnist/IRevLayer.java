package com.example.mnist;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.deeplearning4j.nn.conf.dropout.Dropout;

public class IRevLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected boolean first;
    protected int pad;
    protected int stride;
    protected InjectivePad InjPad;
    protected Psi psi;



    public IRevLayer(NeuralNetConfiguration conf) {
        super(conf);
        first = ((IRevLayer) conf().getLayer()).getfirst();
        pad = ((IRevLayer) conf().getLayer()).getOutCh() * 2 - ((IRevLayer) conf().getLayer()).getInCh();
        stride = ((IRevLayer) conf().getLayer()).getStride();
        InjPad = new InjectivePad(pad, conf);
        psi = new Psi(conf, stride);
        int in_ch = ((IRevLayer) conf().getLayer()).getInCh();
        int out_ch = ((IRevLayer) conf().getLayer()).getOutCh();
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
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {

    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}

