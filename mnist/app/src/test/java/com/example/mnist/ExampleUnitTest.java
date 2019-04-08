package com.example.mnist;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    @Test
    public void addition_isCorrect() {
        assertEquals(4, 2 + 2);
    }

    @Test
    public void IRevLayer_forward_isCorrect() {
        INDArray secondArray = Nd4j.ones(new int[]{10,10,3});

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .list()
                .layer(new IRevLayer.Builder()
                        .First(false)
                        .Mult(4)
                        .nIn(1)
                        .nOut(2)
                        .Stride(1)
                        .AffineBN(true)
                        .build())
                .build();
        MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
        INDArray output = myNetwork.output(secondArray);
        System.out.print(output);

    }
}


