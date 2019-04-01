package com.example.mnist;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import java.io.File;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

import android.os.Environment;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.util.Log;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class InjectivePad extends BaseLayer<ConvolutionLayer> {
    protected int padSize;
    protected ZeroPaddingLayer pad;

    public InjectivePad(int PadSize, NeuralNetConfiguration conf){
        super(conf);
        padSize = PadSize;
        pad = new ZeroPaddingLayer(0, padSize, 0, 0);

    }
    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        input = input.permute(0, 2, 1, 3);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new ZeroPaddingLayer.Builder(0, padSize, 0, 0)
                        .build())
                .setInputType(InputType.convolutionalFlat(input.shape()[0],input.shape()[1], input.shape()[2])) // InputType.convolutional for normal image
                .build();
        MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
        myNetwork.init();
        INDArray output = myNetwork.output(input);
        return output.permute(0, 2, 1, 3);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

}
