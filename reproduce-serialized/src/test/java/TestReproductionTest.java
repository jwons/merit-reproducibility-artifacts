import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import libsvm.svm_model;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationModel;
import org.tribuo.classification.sgd.linear.LinearSGDModel;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.common.tree.Node;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.ensemble.EnsembleModel;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.liblinear.LibLinearRegressionModel;
import org.tribuo.regression.rtree.IndependentRegressionTreeModel;
import org.tribuo.regression.sgd.fm.FMRegressionModel;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.slm.SLMTrainer;
import org.tribuo.regression.slm.SparseLinearModel;
import org.tribuo.reproducibility.ReproUtil;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestReproductionTest {

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class, LinearSGDTrainer.class, BaggingTrainer.class, SLMTrainer.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    @Test
    public void testRegression() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());
        System.out.println(directoryPath);
        String provFiles[] = directoryPath.list();

        for(int i=0; i<provFiles.length; i++) {
            HashMap<String, Evaluation> evals = TestReproduction.reproduceModel(provFiles[i],
                    uModels.getFile(),
                    uData.getFile(),
                    new RegressionEvaluator());

            Assertions.assertEquals(evals.get("old").asMap(), evals.get("new").asMap());
        }
    }

    @Test
    public void testClassification() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());
        System.out.println(directoryPath);
        String provFiles[] = directoryPath.list();

        for(int i=0; i<provFiles.length; i++) {
            HashMap<String, Evaluation> evals = TestReproduction.reproduceModel(provFiles[i],
                    uModels.getFile(),
                    uData.getFile(),
                    new LabelEvaluator());

            Assertions.assertEquals(evals.get("old").asMap(), evals.get("new").asMap());
        }
    }

    @Test
    public void testLinearWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/logistic.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        LabelEvaluator labelEvaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = labelEvaluator.evaluate(oldModel,splitter.getTest());
        LabelEvaluation newEvaluation = labelEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());
        DenseMatrix oldWeights = ((LinearSGDModel) oldModel).getWeightsCopy();
        DenseMatrix newWeights = ((LinearSGDModel) newModel).getWeightsCopy();

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        for (int i = 0; i < oldWeights.getDimension1Size(); i++){
            for (int j = 0; j < oldWeights.getDimension2Size(); j++){
                Assertions.assertEquals(oldWeights.get(i, j), newWeights.get(i, j));
            }
        }
    }

    @Test
    public void testElasticNetWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/enet.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());
        Map<String, SparseVector> oldWeights = ((SparseLinearModel) oldModel).getWeights();
        Map<String, SparseVector> newWeights = ((SparseLinearModel) newModel).getWeights();

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        TreeSet<String> mapAkeys = new TreeSet<>(oldWeights.keySet());
        TreeSet<String> mapBkeys = new TreeSet<>(newWeights.keySet());
        TreeSet<String> intersectionOfKeys = new TreeSet<>(oldWeights.keySet());

        intersectionOfKeys.retainAll(mapBkeys);

        Assertions.assertEquals(mapAkeys.size(), mapBkeys.size());

        for (String key : intersectionOfKeys){
            Assertions.assertEquals(oldWeights.get(key), newWeights.get(key));
        }
    }

    @Test
    public void testLarsWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/lars.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());
        Map<String, SparseVector> oldWeights = ((SparseLinearModel) oldModel).getWeights();
        Map<String, SparseVector> newWeights = ((SparseLinearModel) newModel).getWeights();

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        TreeSet<String> mapAkeys = new TreeSet<>(oldWeights.keySet());
        TreeSet<String> mapBkeys = new TreeSet<>(newWeights.keySet());
        TreeSet<String> intersectionOfKeys = new TreeSet<>(oldWeights.keySet());

        intersectionOfKeys.retainAll(mapBkeys);

        Assertions.assertEquals(mapAkeys.size(), mapBkeys.size());

        for (String key : intersectionOfKeys){
            Assertions.assertEquals(oldWeights.get(key), newWeights.get(key));
        }
    }

    @Test
    public void testFMRegressionWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/fm.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());
        DenseMatrix oldWeights = ((FMRegressionModel) oldModel).getLinearWeightsCopy();
        DenseMatrix newWeights = ((FMRegressionModel) newModel).getLinearWeightsCopy();

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        for (int i = 0; i < oldWeights.getDimension1Size(); i++){
            for (int j = 0; j < oldWeights.getDimension2Size(); j++){
                Assertions.assertEquals(oldWeights.get(i, j), newWeights.get(i, j));
            }
        }

        Tensor[] oldFactors = ((AbstractFMModel) oldModel).getFactorsCopy();
        Tensor[] newFactors = ((AbstractFMModel) newModel).getFactorsCopy();
        for (int k = 0; k < oldFactors.length; k++) {
            DenseMatrix oldFactorWeights = (DenseMatrix) oldFactors[k];
            DenseMatrix newFactorWeights = (DenseMatrix) newFactors[k];
            for (int i = 0; i < oldWeights.getDimension1Size(); i++){
                for (int j = 0; j < oldWeights.getDimension2Size(); j++){
                    Assertions.assertEquals(oldFactorWeights.get(i, j), newFactorWeights.get(i, j));
                }
            }
        }
    }

    @Test
    public void testFMClassificationWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/fm.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        LabelEvaluator labelEvaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = labelEvaluator.evaluate(oldModel,splitter.getTest());
        LabelEvaluation newEvaluation = labelEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());
        DenseMatrix oldWeights = ((FMClassificationModel) oldModel).getLinearWeightsCopy();
        DenseMatrix newWeights = ((FMClassificationModel) newModel).getLinearWeightsCopy();

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        for (int i = 0; i < oldWeights.getDimension1Size(); i++){
            for (int j = 0; j < oldWeights.getDimension2Size(); j++){
                Assertions.assertEquals(oldWeights.get(i, j), newWeights.get(i, j));
            }
        }

        Tensor[] oldFactors = ((AbstractFMModel) oldModel).getFactorsCopy();
        Tensor[] newFactors = ((AbstractFMModel) newModel).getFactorsCopy();
        for (int k = 0; k < oldFactors.length; k++) {
            DenseMatrix oldFactorWeights = (DenseMatrix) oldFactors[k];
            DenseMatrix newFactorWeights = (DenseMatrix) newFactors[k];
            for (int i = 0; i < oldWeights.getDimension1Size(); i++){
                for (int j = 0; j < oldWeights.getDimension2Size(); j++){
                    Assertions.assertEquals(oldFactorWeights.get(i, j), newFactorWeights.get(i, j));
                }
            }
        }
    }

    @Test
    public void testLibSVMRegressionWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(directoryPath + "/libsvm.model"))){
            oldModel = (Model) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        List<svm_model> oldModels = ((LibSVMModel<Regressor>) oldModel).getInnerModels();
        List<svm_model> newModels = ((LibSVMModel<Regressor>) newModel).getInnerModels();

        for (int i = 0; i < oldModels.size(); i++) {
            Assertions.assertTrue(LibSVMModel.modelEquals(oldModels.get(i),newModels.get(i)));
        }
    }

    @Test
    public void testLibSVMClassificationWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(directoryPath + "/libsvm.model"))){
            oldModel = (Model) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        LabelEvaluator labelEvaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = labelEvaluator.evaluate(oldModel,splitter.getTest());
        LabelEvaluation newEvaluation = labelEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        List<svm_model> oldModels = ((LibSVMModel<Label>) oldModel).getInnerModels();
        List<svm_model> newModels = ((LibSVMModel<Label>) newModel).getInnerModels();

        for (int i = 0; i < oldModels.size(); i++) {
            Assertions.assertTrue(LibSVMModel.modelEquals(oldModels.get(i),newModels.get(i)));
        }
    }

    @Test
    public void testLibLinearRegressionWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/liblinear.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        Assertions.assertEquals(((LibLinearRegressionModel)oldModel).getTopFeatures(-1), ((LibLinearRegressionModel)oldModel).getTopFeatures(-1));
    }

    @Test
    public void testLibLinearClassificationWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/liblinear.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        LabelEvaluator labelEvaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = labelEvaluator.evaluate(oldModel,splitter.getTest());
        LabelEvaluation newEvaluation = labelEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        Assertions.assertEquals(((LibLinearModel)oldModel).getTopFeatures(-1), ((LibLinearModel)oldModel).getTopFeatures(-1));

    }

    @Test
    public void testCARTRegressionWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/joint-cart-reg.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        Assertions.assertEquals(((TreeModel)oldModel).getRoot(), ((TreeModel)oldModel).getRoot());
    }

    @Test
    public void testCARTClassificationWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try {
            FileInputStream fileIn = new FileInputStream(directoryPath + "/cart.model");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            oldModel = (Model) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        LabelEvaluator labelEvaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = labelEvaluator.evaluate(oldModel,splitter.getTest());
        LabelEvaluation newEvaluation = labelEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        Assertions.assertEquals(((TreeModel)oldModel).getRoot(), ((TreeModel)oldModel).getRoot());
    }

    @Test
    public void testRFClassificationWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(directoryPath + "/rf.model"))) {
            oldModel = (Model) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        compareClassificationTreeEnsemble(oldModel,uData);
    }

    @Test
    public void testExtraClassificationWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/classification");
        URL uData = TestReproductionTest.class.getResource("/data/bezdekIris.data");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(directoryPath + "/extra.model"))) {
            oldModel = (Model) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        compareClassificationTreeEnsemble(oldModel,uData);
    }

    @Test
    public void testRFRegressionWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(directoryPath + "/rf-reg.model"))) {
            oldModel = (Model) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        compareRegressionTreeEnsemble(oldModel,uData);
    }

    @Test
    public void testExtraRegressionWeights() throws Exception {
        URL uModels = TestReproductionTest.class.getResource("/models/regression");
        URL uData = TestReproductionTest.class.getResource("/data/winequality-red.csv");
        File directoryPath = new File(uModels.getFile());

        Model oldModel = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(directoryPath + "/extra-reg.model"))) {
            oldModel = (Model) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("Model class not found");
            c.printStackTrace();
        }

        compareRegressionTreeEnsemble(oldModel,uData);
    }

    private static void compareClassificationTreeEnsemble(Model oldModel, URL uData) throws Exception {
        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        LabelEvaluator labelEvaluator = new LabelEvaluator();

        LabelEvaluation oldEvaluation = labelEvaluator.evaluate(oldModel,splitter.getTest());
        LabelEvaluation newEvaluation = labelEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        System.out.println(ReproUtil.diffProvenance(oldModel.getProvenance(), newModel.getProvenance()));

        List<Model<Label>> oldModels = ((EnsembleModel<Label>)oldModel).getModels();
        List<Model<Label>> newModels = ((EnsembleModel<Label>)newModel).getModels();
        for (int i = 0; i < oldModels.size(); i++) {
            Node<Label> oldRoot = ((TreeModel<Label>)oldModels.get(i)).getRoot();
            Node<Label> newRoot = ((TreeModel<Label>)newModels.get(i)).getRoot();
            Assertions.assertEquals(oldRoot, newRoot);
        }
    }

    private static void compareRegressionTreeEnsemble(Model oldModel, URL uData) throws Exception {
        ReproUtil repro = new ReproUtil(oldModel);
        ArrayList<String> componentNames = new ArrayList<String>(repro.getConfigurationManager().getComponentNames());
        String sourceKey = null;
        for (String name : componentNames){
            if(name.length() > 13 && "csvdatasource".equals(name.substring(0, 13))){
                sourceKey = name;
            }
        }
        repro.getConfigurationManager().overrideConfigurableProperty(sourceKey, "dataPath", new SimpleProperty(uData.getFile()));
        Model newModel = repro.reproduceFromProvenance();

        var splitter = new TrainTestSplitter((DataSource) repro.getConfigurationManager().lookup(sourceKey), 0.7f, 0L);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();

        RegressionEvaluation oldEvaluation = regressionEvaluator.evaluate(oldModel,splitter.getTest());
        RegressionEvaluation newEvaluation = regressionEvaluator.evaluate(newModel,splitter.getTest());

        Assertions.assertEquals(oldEvaluation.asMap(), newEvaluation.asMap());

        List<Model<Regressor>> oldModels = ((EnsembleModel<Regressor>)oldModel).getModels();
        List<Model<Regressor>> newModels = ((EnsembleModel<Regressor>)newModel).getModels();
        for (int i = 0; i < oldModels.size(); i++) {
            Map<String,Node<Regressor>> oldRoot = ((IndependentRegressionTreeModel)oldModels.get(i)).getRoots();
            Map<String,Node<Regressor>> newRoot = ((IndependentRegressionTreeModel)newModels.get(i)).getRoots();
            Assertions.assertEquals(oldRoot, newRoot);
        }
    }

}
