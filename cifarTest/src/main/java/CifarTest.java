import javafx.util.Pair;
import org.datavec.image.transform.*;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

public class CifarTest {
    protected static final Logger log = LoggerFactory.getLogger(CifarTest.class);
    private static final String basePath = System.getProperty("user.home") + "/cifar";
    private static final boolean manual_gradients = false;
    private static final boolean half_precision = false;
    private static final boolean microbatch = false;

    public static void main(String[] args) {
        try {
            int[] nChannels = new int[]{16, 64, 256};
            int[] nBlocks = new int[]{2, 2, 2};
            int[] nStrides = new int[]{1, 2, 2};
            int channels = 3;
            int init_ds = 0;
            int in_ch = channels * (int) Math.pow(2, init_ds);
            int n = in_ch / 2;
            int outputNum = 10; // number of output classes
            final int numRows = 32;
            final int numColumns = 32;
            int rngSeed = 1234; // random number seed for reproducibility
            int numEpochs = 200; // number of epochs to perform
            int batchSize = 128;
            int microBatchSize = 16;
            int microNum = batchSize / microBatchSize;
            int mult = 4;
            double init_lr = 0.1;
            Random randNumGen = new Random(rngSeed);

            Map<Integer, Double> learningRateSchedule = new HashMap<>();
            learningRateSchedule.put(0, init_lr);
            learningRateSchedule.put(60, init_lr * Math.pow(0.2, 1));
            learningRateSchedule.put(120, init_lr * Math.pow(0.2, 2));
            learningRateSchedule.put(160, init_lr * Math.pow(0.2, 3));

            File baseDir = new File(basePath);
            if (!baseDir.exists()) {
                baseDir.mkdirs();
            }
            DL4JResources.setBaseDirectory(baseDir);
            List<org.nd4j.linalg.primitives.Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
            pipeline.add(new org.nd4j.linalg.primitives.Pair<>(new BoxImageTransform(40, 40), 1.0));
            pipeline.add(new org.nd4j.linalg.primitives.Pair<>(new CropImageTransform(randNumGen, 4), 1.0));
            pipeline.add(new org.nd4j.linalg.primitives.Pair<>(new FlipImageTransform(1), 0.5));
            ImageTransform transform = new PipelineImageTransform(pipeline, false); // pad to 40*40, then crop to 32*32, then flip horizontally
            Cifar10DataSetIterator cifarTrain = new Cifar10DataSetIterator(batchSize, new int[]{numRows, numColumns},
                DataSetType.TRAIN, transform, rngSeed);
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(cifarTrain);
            cifarTrain.setPreProcessor(normalizer);
            Cifar10DataSetIterator cifarTest = new Cifar10DataSetIterator(batchSize, new int[]{numRows, numColumns},
                DataSetType.TEST, null, rngSeed);
            cifarTest.setPreProcessor(normalizer);

            if (half_precision) {
                Nd4j.setDefaultDataTypes(DataType.HALF, DataType.HALF);
            }

            NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new Nesterovs(new MapSchedule(ScheduleType.EPOCH,
                    learningRateSchedule), 0.9))
                .weightDecay(5e-4);

            if (half_precision) {
                config.dataType(DataType.HALF);
            }

            ComputationGraphConfiguration.GraphBuilder graph = config.graphBuilder();

            graph.addInputs("input").setInputTypes(InputType.convolutional(numRows, numColumns, channels));
            String lastLayer = "input";
            if (init_ds != 0) {
                graph.addLayer("init_psi", new PsiLayer.Builder()
                    .BlockSize(init_ds)
                    .build(), "input");
                lastLayer = "init_psi";
            }

            graph.addVertex("x0", new SubsetVertexN(0, n - 1), lastLayer)
                .addVertex("tilde_x0", new SubsetVertexN(n, in_ch - 1), lastLayer);
            int in_ch_Block = in_ch;
            int inputH = numRows;
            int inputW = numColumns;
            String input1 = "x0";
            String input2 = "tilde_x0";
            boolean first = true;
            List<IRevBlock> blockList = new ArrayList<>();
            for (int i = 0; i < nBlocks.length; i++) { // for each stage
                for (int j = 0; j < nBlocks[i]; j++) { // for each block in the stage
                    int stride = 1;
                    if (j == 0) {
                        stride = nStrides[i];
                    }
                    IRevBlock innerIRevBlock = new IRevBlock(graph, batchSize, inputH, inputW, in_ch_Block,
                            nChannels[i], stride, first, mult, input1, input2, String.valueOf(i) + j);
                    String[] outputs = innerIRevBlock.getOutput();
                    input1 = outputs[0];
                    input2 = outputs[1];
                    blockList.add(innerIRevBlock);
                    in_ch_Block = 2 * nChannels[i];
                    inputH = innerIRevBlock.getOutputH();
                    inputW = innerIRevBlock.getOutputW();
                    first = false;
                }
            }

            ProbLayer probLayer = new ProbLayer(batchSize, inputH, inputW,nChannels[nChannels.length - 1] * 2, outputNum);
            LossLayer lossLayer = new LossLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .build();

            graph.addVertex("merge", new MergeVertex(), input1, input2)
                .addLayer("outputProb", probLayer,"merge")
                .addLayer("output", lossLayer, "outputProb")
                .setOutputs("output");


            ComputationGraphConfiguration conf = graph.build();
            ComputationGraph model = new ComputationGraph(conf);
            model.init();
            MemoryManager mg = Nd4j.getMemoryManager();
            mg.togglePeriodicGc(true);

            log.info("start training");
            if (manual_gradients) {
                if (microbatch) {
                    int i = 0;
                    ComputationGraphUpdater updater = model.getUpdater();
                    for (int epoch = 0; epoch < numEpochs; epoch++) {
                        while (cifarTrain.hasNext()) {
                            boolean flag = true; // If it is the first time, assign gradients; otherwise, accumulate gradients.
                            log.info("Running iter " + i);
                            DataSet data = cifarTrain.next();
                            List<DataSet> microbatch = data.batchBy(microBatchSize);
                            Iterator<DataSet> microItor = microbatch.iterator();
                            Gradient gradient = new DefaultGradient();
                            while (microItor.hasNext()){
                                // fetch data
                                DataSet microdata = microItor.next();
                                INDArray microlabel = microdata.getLabels();
                                INDArray microfeatures = microdata.getFeatures();

                                // Forward Pass
                                INDArray[] microOutputs = model.output(false, false, microfeatures);
                                INDArray merge = microOutputs[1];

                                // Backward Pass
                                INDArray[] outputGradients = probLayer.gradient(merge, microlabel);
                                INDArray dwGradient = outputGradients[1];
                                INDArray dbGradient = outputGradients[2];
                                INDArray[] lossGradient = Utils.splitHalf(outputGradients[0]);
                                INDArray[] hiddens = Utils.splitHalf(merge);
                                Pair<String, INDArray>[] grads = computeGradient(gradient, hiddens[0], hiddens[1],
                                        nBlocks, blockList, lossGradient);
                                if (flag) {
                                    gradient.setGradientFor("outputProb_denseWeight", dwGradient.div(microNum), 'c');
                                    gradient.setGradientFor("outputProb_denseBias", dbGradient.div(microNum));
                                    for (Pair<String, INDArray> grad : grads) {
                                        gradient.setGradientFor(grad.getKey(), grad.getValue(), 'c');
                                    }
                                    flag = false;
                                } else{
                                    gradient.setGradientFor("outputProb_denseWeight", gradient.getGradientFor("outputProb_denseWeight").add(dwGradient.div(microNum)), 'c');
                                    gradient.setGradientFor("outputProb_denseBias", gradient.getGradientFor("outputProb_denseBias").add(dbGradient.div(microNum)));
                                    for (Pair<String, INDArray> grad : grads) {
                                        gradient.setGradientFor(grad.getKey(), gradient.getGradientFor(grad.getKey()).add(grad.getValue().div(microNum)), 'c');
                                    }
                                }
                            }

                            updater.update(gradient, i, epoch, batchSize, LayerWorkspaceMgr.noWorkspaces());
                            model.params().subi(gradient.gradient());

                            // Evaluate test accuracy per 50 iters
                            if (i % 50 == 0) {
                                log.info("Evaluate model....");
                                Evaluation evalTest = new Evaluation(outputNum);
                                while(cifarTest.hasNext()){
                                    DataSet next = cifarTest.next();
                                    INDArray out = model.output(false, false, next.getFeatures())[0];
                                    evalTest.eval(next.getLabels(), out);
                                }
                                cifarTest.reset();
                                log.info(evalTest.stats());
                            }
                            // Evaluate training accuracy for each iter
                            INDArray label = data.getLabels();
                            INDArray features = data.getFeatures();
                            INDArray[] microOutputs = model.output(false, false, features);
                            INDArray output = microOutputs[0];
                            Evaluation eval = new Evaluation(outputNum);
                            eval.eval(label, output);
                            log.info(eval.stats());
                            i++;
                        }
                        cifarTrain.reset();
                    }
                } else {
                    int i = 0;
                    ComputationGraphUpdater updater = model.getUpdater();
                    for (int epoch = 0; epoch < numEpochs; epoch++) {
                        while (cifarTrain.hasNext()) {
                            log.info("Running iter " + i);
                            DataSet data = cifarTrain.next();
                            INDArray label = data.getLabels();
                            INDArray features = data.getFeatures();

                            // Forward Pass
                            INDArray[] outputs = model.output(false, false, features);
                            INDArray output = outputs[0];
                            INDArray merge = outputs[1];

                            // Backward Pass
                            Gradient gradient = new DefaultGradient();
                            INDArray[] outputGradients = probLayer.gradient(merge, label);
                            INDArray dwGradient = outputGradients[1];
                            INDArray dbGradient = outputGradients[2];
                            INDArray[] lossGradient = Utils.splitHalf(outputGradients[0]);
                            INDArray[] hiddens = Utils.splitHalf(merge);
                            Pair<String, INDArray>[] grads = computeGradient(gradient, hiddens[0], hiddens[1],
                                    nBlocks, blockList, lossGradient);
                            for (Pair<String, INDArray> grad : grads) {
                                gradient.setGradientFor(grad.getKey(), grad.getValue(), 'c');
                            }
                            gradient.setGradientFor("outputProb_denseWeight", dwGradient, 'c');
                            gradient.setGradientFor("outputProb_denseBias", dbGradient);
                            updater.update(gradient, i, epoch, batchSize, LayerWorkspaceMgr.noWorkspaces());
                            model.params().subi(gradient.gradient());

                            // Evaluation
                            if (i % 50 == 0) {
                                log.info("Evaluate model....");
                                Evaluation evalTest = new Evaluation(outputNum);
                                while(cifarTest.hasNext()){
                                    DataSet next = cifarTest.next();
                                    INDArray out = model.output(false, false, next.getFeatures())[0];
                                    evalTest.eval(next.getLabels(), out);
                                }
                                cifarTest.reset();
                                log.info(evalTest.stats());
                            }
                            Evaluation eval = new Evaluation(outputNum);
                            eval.eval(label, output);
                            log.info(eval.stats());
                            i++;
                        }
                        cifarTrain.reset();
                    }
                }
            } else {
                int i = 0;
                for (int epoch = 0; epoch < numEpochs; epoch++) {
                    log.info("Model fit Epoch " + epoch);
                    while (cifarTrain.hasNext()) {
                        log.info("Model fit Iter " + i);
                        DataSet data = cifarTrain.next();
                        INDArray[] labels = new INDArray[]{data.getLabels()};
                        INDArray[] features = new INDArray[]{data.getFeatures()};
                        // Masks should be null so no appending
                        INDArray[] featureMasks = new INDArray[]{data.getFeaturesMaskArray()};
                        INDArray[] labelMasks = new INDArray[]{data.getLabelsMaskArray()};
                        model.fit(features, labels, featureMasks, labelMasks);
                        log.info("Model fit Score " + model.score());
                        i++;
                    }
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    // This function computes the total gradient of the graph without referring to the stored activation
    protected static Pair<String, INDArray>[] computeGradient(Gradient gradient, INDArray y1, INDArray y2, int[] nBlocks,
                                          List<IRevBlock> blockList, INDArray[] lossGradient) {
        INDArray dy1 = lossGradient[0];
        INDArray dy2 = lossGradient[1];
        int cnt = blockList.size() - 1;
        Pair<String, INDArray>[] grads = new Pair[blockList.size() * 3];
        // from the last layer to the first layer
        for (int i = nBlocks.length - 1; i >= 0; i -= 1) { // for each stage
            for (int j = nBlocks[i] - 1; j >= 0; j -= 1) { // for each iRevBlock
                IRevBlock iRev = blockList.get(cnt);
                INDArray[] x = iRev.inverse(y1, y2);
                INDArray x1 = x[0];
                INDArray x2 = x[1];
                Runtime.getRuntime().gc();
                y1 = x1;
                y2 = x2;
                List<INDArray> gradients = iRev.gradient(x2, dy1, dy2);
                String prefix = iRev.getPrefix();
                dy1 = gradients.get(0);
                dy2 = gradients.get(1);
                grads[(cnt + 1) * 3 - 3] = new Pair(prefix + "btnk_conv1Weight", gradients.get(2));
                grads[(cnt + 1) * 3 - 2] = new Pair(prefix + "btnk_conv2Weight", gradients.get(3));
                grads[(cnt + 1) * 3 - 1] = new Pair(prefix + "btnk_conv3Weight", gradients.get(4));
                cnt -= 1;
            }
        }
        return grads;
    }
}
