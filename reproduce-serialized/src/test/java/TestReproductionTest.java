import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LinearSGDModel;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.slm.SLMTrainer;
import org.tribuo.reproducibility.ReproUtil;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
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
}
