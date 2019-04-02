package com.example.mnist;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Psi extends BaseLayer<ConvolutionLayer> {
    long blockSize = 0;
    long blockSizeSq = 0;

    /**
     * Constructor: initialization
     * @param conf
     * @param blockSize
     */
    public Psi(NeuralNetConfiguration conf, int blockSize) {
        super(conf);
        this.blockSize = blockSize;
        this.blockSizeSq = blockSize*blockSize;
    }

    /**
     * inverse
     * @param input
     * @return INDArray
     */
    /*
    public INDArray inverse(INDArray input) {

        INDArray output = input.permute(0, 2, 3, 1);
        long[] shape = output.shape();
        System.out.print(shape);
        long batchSize = shape[0];
        long dHeight = shape[1];
        long dWidth = shape[2];
        long dDepth = shape[3];
        long sDepth = dDepth / this.blockSizeSq;
        long sWidth = dWidth * this.blockSize;
        long sHeight = dHeight * this.blockSize;
        long[] shapeParam = new long[]{batchSize, dHeight, dWidth, this.blockSizeSq, sDepth};
        INDArray t1 = output.reshape(shapeParam);
        t1.





    }*/

    /**
     * inverse
     * @param input
     * @return INDArray
     *         output = input.permute(0, 2, 3, 1)
     *         (batch_size, s_height, s_width, s_depth) = output.size()
     *         d_depth = s_depth * self.block_size_sq
     *         d_height = int(s_height / self.block_size)
     *
     *         t_1 = output.split(self.block_size, 2)
     *         stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
     *         output = torch.stack(stack, 1)
     *
     *         output = output.permute(0, 2, 1, 3)
     *         output = output.permute(0, 3, 1, 2)
     */
    public INDArray forward(INDArray input) {
        INDArray output = input.permute(0, 2, 3, 1);
        long[] shape = output.shape();
        System.out.print(shape);
        assert shape.length >= 4;
        long batchSize = shape[0];
        long sHeight = shape[1];
        long sWidth = shape[2];
        long sDepth = shape[3];
        long dDepth = sDepth * this.blockSizeSq;
        long dHeight = sHeight / this.blockSize;

        for (int i = 0; i < sWidth; i += this.blockSize) {
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
